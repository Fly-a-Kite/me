import json
from collections import Counter

from datadiff import methodology_report, reporter
from datadiff.methodology_report import write_methodology_report
from datadiff.util import append_jsonl, dump_json, run_meta_path


def _write_run(path, *, findings_by_index, artifact_dir=None, run_provenance=None):
    for idx in range(2):
        findings = findings_by_index.get(idx, [])
        row = {
            "status": "bug" if findings else "ok",
            "case_index": idx,
            "elapsed_s": (idx + 1) * 0.1,
            "case": {"case_id": f"case-{path.stem}-{idx}", "seed": idx, "program": {"operations": []}},
            "findings": findings,
            "behavior_signature": f"sig-{path.stem}-{idx}",
            "backend_status": {},
            "quality_oracles": [
                {"name": "mutation", "verdict": "productive_mutation", "passed": True, "score": 1.5},
                {"name": "feedback", "verdict": "finding_yield" if findings else "new_behavior_yield", "passed": True, "score": 1.0},
                {"name": "guidance", "verdict": "guided_productive" if idx == 0 else "guided_redundant", "passed": idx == 0, "score": 3.0},
            ],
            "guidance": {
                "score": 3.0 - idx,
                "features": ["semantic_family:join_membership", "semantic_signal:topk_filter_pushdown"],
                "matched_targets": ["groupby"] if idx == 0 else [],
                "matched_semantic_targets": ["groupby"] if idx == 0 else [],
                "discovery_bias_hits": ["groupby|semantic_signal:"] if idx == 0 else [],
                "matched_semantic_target_count": 1 if idx == 0 else 0,
                "discovery_bias_hit_count": 1 if idx == 0 else 0,
                "candidate_count": 2,
                "contributing_candidate_count": 1,
                "pruned_candidate_count": 0,
                "feature_count": 2,
                "frontier_bucket_count": 1,
                "discovery_bucket_count": 2,
                "data_sensitivity": 0.5,
                "path_coverage_proxy": 0.25,
                "frontier_conformance": 0.75,
                "discovery_diversity_bonus": 0.1,
                "candidate_pool_diversity_bonus": 0.05,
                "discovery_stale_penalty": -0.2,
                "contribution_potential": 1.0,
            },
            "is_new_behavior": idx == 0,
            "signal_new_behavior": idx == 0 and bool(findings),
            "candidate_source": "feedback_mutation" if idx == 0 else "generated",
            "stored_in_feedback_corpus": idx == 0,
            "feedback_summary": {
                "candidate_source": "feedback_mutation" if idx == 0 else "generated",
                "stored_in_feedback_corpus": idx == 0,
                "quality_oracle_count": 3,
                "quality_pass_count": 3 if idx == 0 else 2,
                "quality_fail_count": 0 if idx == 0 else 1,
                "quality_score_total": 5.5 if idx == 0 else 2.5,
                "source_reward_adjustment": 0.4 if idx == 0 else -0.05,
                "guidance_reward_adjustment": 0.25 if idx == 0 else -0.05,
                "seed_schedule_delta": 3.6 if idx == 0 else 0.3,
                "mutation_oracle_verdict": "productive_mutation" if idx == 0 else "",
                "feedback_oracle_verdict": "finding_yield" if findings else "new_behavior_yield",
                "guidance_oracle_verdict": "guided_productive" if idx == 0 else "guided_redundant",
            },
        }
        if artifact_dir and findings:
            row["bug_dir"] = str(artifact_dir)
        append_jsonl(row, path)
    dump_json(
        {
            "elapsed_s": 0.2,
            "throughput_cases_s": 10.0,
            "backends": [],
            "targets": [],
            "common_capabilities": [],
            "run_provenance": run_provenance or {},
            "config": {
                "guidance_targets": ["groupby", "topk"],
                "effective_guidance_targets": [
                    "groupby",
                    "topk",
                    "semantic_family:join_membership",
                    "semantic_signal:topk_filter_pushdown",
                ],
                "semantic_focus_families": ["join_membership"],
                "semantic_focus_signals": ["topk_filter_pushdown"],
                "discovery_biases": [
                    {
                        "targets": ["groupby"],
                        "feature_prefixes": ["semantic_signal:"],
                        "keep_in_pool": True,
                    }
                ],
            },
        },
        run_meta_path(path),
    )


def test_write_methodology_report_links_evidence_chain_and_space_metrics(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    freeze_dir = reports_dir / "freeze"
    artifact_dir = tmp_path / "bugs" / "bug-candidate"
    freeze_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "evidence.json").write_text('{"ok": true}\n', encoding="utf-8")
    for name in ("case.json", "results.json", "normalized.json", "findings.json", "config.json"):
        (artifact_dir / name).write_text("{}\n", encoding="utf-8")
    (artifact_dir / "reproduce.py").write_text("print('repro')\n", encoding="utf-8")
    (artifact_dir / "reduced_case.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "reproduce_reduced.py").write_text("print('reduced')\n", encoding="utf-8")
    (artifact_dir / "triage.json").write_text("{}\n", encoding="utf-8")
    (artifact_dir / "standalone_polars_reverse_division_columns.py").write_text(
        "print('standalone')\n",
        encoding="utf-8",
    )
    issue_bundle_dir = tmp_path / "new_issue" / "generated" / "issue-bundles"
    issue_bundle_dir.mkdir(parents=True)
    candidate_pipeline_dir = tmp_path / "new_issue" / "generated" / "candidate-pipelines" / "pipeline-one"
    candidate_pipeline_dir.mkdir(parents=True)
    dump_json(
        {
            "schema_version": "issue-bundle-v1",
            "generated_at": "2026-05-28T11:22:01Z",
            "generated_by": "datadiff issue-bundle",
            "inputs": {"run_reproducers": True, "repeat_count": 2},
            "issues": [
                {
                    "issue_path": "new_issue/ready.md",
                    "reproducer_path": "new_issue/generated/issue-bundles/reproducers/ready.py",
                    "run": {"returncode": 0, "timed_out": False},
                },
                {
                    "issue_path": "new_issue/dedup.md",
                    "reproducer_path": "new_issue/generated/issue-bundles/reproducers/dedup.py",
                    "run": {"returncode": 0, "timed_out": False},
                },
            ],
            "summary": {
                "family_count": 2,
                "families": ["ready_family@polars", "dedup_family@duckdb"],
                "issue_count": 2,
                "extracted_reproducer_count": 2,
                "missing_reproducer_count": 0,
                "compile_failure_count": 0,
                "executed_reproducer_count": 2,
                "executed_reproducer_attempt_count": 4,
                "flaky_reproducer_count": 0,
                "nonzero_exit_count": 0,
                "nonzero_exit_attempt_count": 0,
                "timeout_count": 0,
                "timeout_attempt_count": 0,
            },
        },
        issue_bundle_dir / "manifest.json",
    )
    dump_json(
        {
            "schema_version": "bug-sprint-v1",
            "generated_at": "2026-05-28T12:00:00Z",
            "scheduler": {
                "lanes": [
                    {
                        "lane_id": "arrow_layout",
                        "score": 2.0,
                        "budget_multiplier": 1.4,
                        "yield_rate": 1.0,
                        "novelty_rate": 0.5,
                        "false_positive_rate": 0.0,
                    }
                ]
            },
            "runs": [
                {
                    "lane_id": "arrow_layout",
                    "status": "completed",
                    "classification": {
                        "fresh_candidate_bug_families": {"fresh_root@polars": 1},
                        "false_positive_reasons": {},
                    },
                    "scheduler": {
                        "score": 2.0,
                        "budget_multiplier": 1.4,
                        "yield_rate": 1.0,
                        "novelty_rate": 0.5,
                        "false_positive_rate": 0.0,
                    },
                }
            ],
        },
        tmp_path / "new_issue" / "generated" / "bug-sprint-report-manifest.json",
    )
    dump_json(
        {
            "schema_version": "candidate-pipeline-v1",
            "strategy_snapshot_path": "new_issue/generated/candidate-pipelines/pipeline-one/strategy-snapshot.json",
            "summary": {
                "candidate_count": 2,
                "rechecked_count": 2,
                "reproduced_count": 1,
                "reduced_count": 1,
                "candidate_bug_verdict_count": 1,
                "issue_draft_count": 1,
                "needs_dedup_check_count": 1,
                "already_submitted_or_confirmed_count": 0,
            },
            "candidates": [
                {
                    "strategy_learning_path": "new_issue/generated/candidate-pipelines/pipeline-one/strategy-learning/candidate-pipeline-learning.json"
                }
            ],
        },
        candidate_pipeline_dir / "manifest.json",
    )
    (candidate_pipeline_dir / "strategy-snapshot.json").write_text("{}", encoding="utf-8")
    learning_dir = candidate_pipeline_dir / "strategy-learning"
    learning_dir.mkdir(parents=True, exist_ok=True)
    (learning_dir / "candidate-pipeline-learning.json").write_text("{}", encoding="utf-8")
    baseline_freeze_manifest = freeze_dir / "baseline-freeze.json"
    baseline_pip_freeze = freeze_dir / "baseline-pip-freeze.txt"
    baseline_git_status = freeze_dir / "baseline-git-status.txt"
    baseline_git_diff = freeze_dir / "baseline-git-diff.patch"
    baseline_launcher_env = freeze_dir / "baseline-launcher-env.txt"
    baseline_strategy_snapshot = freeze_dir / "baseline-strategy-snapshot.json"
    baseline_strategy_learning = freeze_dir / "baseline-strategy-learning.json"
    no_normalizer_freeze_manifest = freeze_dir / "no-normalizer-freeze.json"
    no_normalizer_pip_freeze = freeze_dir / "no-normalizer-pip-freeze.txt"
    no_normalizer_git_status = freeze_dir / "no-normalizer-git-status.txt"
    no_normalizer_launcher_env = freeze_dir / "no-normalizer-launcher-env.txt"
    no_normalizer_strategy_snapshot = freeze_dir / "no-normalizer-strategy-snapshot.json"
    no_normalizer_strategy_learning = freeze_dir / "no-normalizer-strategy-learning.json"
    for path in (
        baseline_freeze_manifest,
        baseline_pip_freeze,
        baseline_git_status,
        baseline_git_diff,
        baseline_launcher_env,
        baseline_strategy_snapshot,
        baseline_strategy_learning,
        no_normalizer_freeze_manifest,
        no_normalizer_pip_freeze,
        no_normalizer_git_status,
        no_normalizer_launcher_env,
        no_normalizer_strategy_snapshot,
        no_normalizer_strategy_learning,
    ):
        path.write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "PROJECT_ROOT", tmp_path)

    baseline_run = runs_dir / "run-baseline.jsonl.gz"
    no_normalizer_run = runs_dir / "run-no-normalizer.jsonl.gz"
    reducer_run = runs_dir / "run-reducer.jsonl.gz"
    _write_run(
        baseline_run,
        findings_by_index={
            1: [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "fresh_root",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["polars"],
                    "signature": "sig-fresh",
                }
            ]
        },
        artifact_dir=artifact_dir,
        run_provenance={
            "vcs": {
                "git_commit": "a" * 40,
                "git_branch": "main",
                "workspace_dirty": False,
            },
            "launch": {
                "source": "tmux",
                "duration": "12h",
            },
            "harness": {
                "authority": True,
                "freeze_intent": True,
                "latest_code_claim": True,
                "evidence_role": "latest_live_authority_12h",
            },
            "freeze_artifacts": {
                "manifest": str(baseline_freeze_manifest),
                "pip_freeze": str(baseline_pip_freeze),
                "git_status": str(baseline_git_status),
                "git_diff": str(baseline_git_diff),
                "launcher_env": str(baseline_launcher_env),
                "strategy_snapshot": str(baseline_strategy_snapshot),
                "strategy_learning": str(baseline_strategy_learning),
            },
        },
    )
    _write_run(
        no_normalizer_run,
        findings_by_index={
            0: [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "normalizer_noise",
                    "triage_verdict": "normalizer_false_positive",
                    "false_positive": True,
                    "signature": "sig-noise",
                }
            ],
            1: [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "nan_inf_semantics",
                    "triage_verdict": "expected_semantic_divergence",
                    "signature": "sig-semantic",
                }
            ]
        },
        run_provenance={
            "vcs": {
                "git_commit": "a" * 40,
                "git_branch": "main",
                "workspace_dirty": False,
            },
            "launch": {
                "source": "tmux",
                "duration": "30m",
            },
            "harness": {
                "authority": False,
                "freeze_intent": False,
                "latest_code_claim": False,
                "evidence_role": "validation_probe",
            },
            "freeze_artifacts": {
                "manifest": str(no_normalizer_freeze_manifest),
                "pip_freeze": str(no_normalizer_pip_freeze),
                "git_status": str(no_normalizer_git_status),
                "launcher_env": str(no_normalizer_launcher_env),
                "strategy_snapshot": str(no_normalizer_strategy_snapshot),
                "strategy_learning": str(no_normalizer_strategy_learning),
            },
        },
    )
    _write_run(
        reducer_run,
        findings_by_index={
            0: [
                {
                    "kind": "semantic_output_mismatch",
                    "root_cause": "known_root",
                    "triage_verdict": "candidate_implementation_bug",
                    "discovery_origin": "issue_replay",
                    "suspicious_backends": ["duckdb"],
                    "signature": "sig-known",
                }
            ]
        },
        artifact_dir=None,
    )
    manifest = runs_dir / "experiment-methodology.json"
    dump_json(
        {
            "presets": ["baseline", "no_normalizer", "reducer"],
            "seeds": [1],
            "target_suite": "core",
            "target_suites": ["core"],
            "evidence_mode": "live",
            "schedule": "adaptive",
            "local_source_scheduler": {"enabled": True, "exploration_weight": 0.2},
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
            },
            "adaptive_state": [
                {
                    "arm_id": "core:baseline:seed1",
                    "target_suite": "core",
                    "pulls": 3,
                    "mean_reward": 4.2,
                    "reward_signal": 3.4,
                    "last_reward": 3.7,
                    "stale_batches": 0,
                },
                {
                    "arm_id": "core:no_normalizer:seed1",
                    "target_suite": "core",
                    "pulls": 2,
                    "mean_reward": 1.1,
                    "reward_signal": 0.4,
                    "last_reward": 0.2,
                    "stale_batches": 2,
                },
            ],
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "baseline",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "baseline",
                    "variant_title": "baseline",
                    "base_preset": "baseline",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "semantic_focus_families": ["join_membership"],
                    "semantic_focus_signals": ["topk_filter_pushdown"],
                    "rq_tags": ["RQ2", "RQ3", "RQ4", "RQ5"],
                    "analysis_tags": ["ablation", "baseline"],
                    "seed": 1,
                    "evidence_mode": "live",
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "no_normalizer",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "no_normalizer",
                    "variant_title": "no_normalizer",
                    "base_preset": "baseline",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2"],
                    "analysis_tags": ["ablation", "noise_control"],
                    "seed": 1,
                    "evidence_mode": "live",
                    "run_file": str(no_normalizer_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "reducer",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "reducer",
                    "variant_title": "reducer",
                    "base_preset": "baseline",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ5"],
                    "analysis_tags": ["ablation", "actionability"],
                    "seed": 1,
                    "evidence_mode": "live",
                    "run_file": str(reducer_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )
    iter_counts = Counter()
    original_iter_jsonl = methodology_report._iter_jsonl

    def counting_iter_jsonl(path):
        iter_counts[str(path)] += 1
        yield from original_iter_jsonl(path)

    monkeypatch.setattr(methodology_report, "_iter_jsonl", counting_iter_jsonl)

    md_path, json_path = write_methodology_report(manifest)

    report = json.loads(json_path.read_text(encoding="utf-8"))
    md = md_path.read_text(encoding="utf-8")
    assert report["schema_version"] == "methodology-report-v1"
    assert report["evidence_chain"]["manifest"] == str(manifest)
    assert report["evidence_chain"]["aggregate_json"].endswith(
        "experiment-summary-experiment-methodology-aggregates.json"
    )
    assert report["coverage"]["matrix_ids"] == ["module_ablation"]
    assert report["coverage"]["comparison_groups"] == ["module_ablation"]
    assert report["coverage"]["variant_ids"] == ["baseline", "no_normalizer", "reducer"]
    assert report["coverage"]["scope_kinds"] == ["core"]
    assert report["coverage"]["oracle_profiles"] == ["differential"]
    assert report["coverage"]["rq_tags"] == ["RQ2", "RQ3", "RQ4", "RQ5"]
    assert report["coverage"]["analysis_tags"] == ["ablation", "actionability", "baseline", "noise_control"]
    assert report["coverage"]["semantic_focus_families"] == ["join_membership"]
    assert report["coverage"]["semantic_focus_signals"] == ["topk_filter_pushdown"]
    assert report["evidence_chain"]["run_logs"] == [str(baseline_run), str(no_normalizer_run), str(reducer_run)]
    assert report["evidence_chain"]["run_provenance_freeze_manifests"] == [
        str(baseline_freeze_manifest),
        str(no_normalizer_freeze_manifest),
    ]
    assert report["evidence_chain"]["run_provenance_pip_freeze_artifacts"] == [
        str(baseline_pip_freeze),
        str(no_normalizer_pip_freeze),
    ]
    assert report["evidence_chain"]["run_provenance_git_status_artifacts"] == [
        str(baseline_git_status),
        str(no_normalizer_git_status),
    ]
    assert report["evidence_chain"]["run_provenance_git_diff_artifacts"] == [str(baseline_git_diff)]
    assert report["evidence_chain"]["run_provenance_launcher_env_artifacts"] == [
        str(baseline_launcher_env),
        str(no_normalizer_launcher_env),
    ]
    assert report["evidence_chain"]["run_provenance_strategy_snapshot_artifacts"] == [
        str(baseline_strategy_snapshot),
        str(no_normalizer_strategy_snapshot),
    ]
    assert report["evidence_chain"]["run_provenance_strategy_learning_artifacts"] == [
        str(baseline_strategy_learning),
        str(no_normalizer_strategy_learning),
    ]
    assert dict(iter_counts) == {
        str(baseline_run): 1,
        str(no_normalizer_run): 1,
        str(reducer_run): 1,
    }
    assert report["run_log_scan"] == {
        "run_logs_missing": 0,
        "run_logs_scan_skipped": False,
        "run_logs_scanned": 3,
        "run_logs_total": 3,
    }
    assert report["coverage"]["evidence_modes"] == {"live": 3}
    assert report["efficiency"]["run_log_bytes"] > 0
    assert report["efficiency"]["artifact_bytes"] > 0
    assert report["efficiency"]["evidence_bytes_per_case"] > 0
    assert "stage_profile" in report["efficiency"]
    assert report["efficiency"]["stage_profile"]["backend_execution_avg_ms"] >= 0.0
    assert report["reproducibility"]["run_provenance"] == {
        "all_live_runs_have_qualified_authority_provenance": False,
        "authority_run_count": 1,
        "clean_workspace_run_count": 2,
        "freeze_artifact_nonempty_counts": {
            "manifest": 2,
            "pip_freeze": 2,
            "git_status": 2,
            "git_diff": 1,
            "launcher_env": 2,
            "strategy_snapshot": 2,
            "strategy_learning": 2,
        },
        "freeze_artifact_present_counts": {
            "manifest": 2,
            "pip_freeze": 2,
            "git_status": 2,
            "git_diff": 1,
            "launcher_env": 2,
            "strategy_snapshot": 2,
            "strategy_learning": 2,
        },
        "freeze_git_diff_paths": [str(baseline_git_diff)],
        "freeze_git_status_paths": [str(baseline_git_status), str(no_normalizer_git_status)],
        "freeze_launcher_env_paths": [str(baseline_launcher_env), str(no_normalizer_launcher_env)],
        "freeze_strategy_snapshot_paths": [str(baseline_strategy_snapshot), str(no_normalizer_strategy_snapshot)],
        "freeze_strategy_learning_paths": [str(baseline_strategy_learning), str(no_normalizer_strategy_learning)],
        "freeze_manifest_paths": [str(baseline_freeze_manifest), str(no_normalizer_freeze_manifest)],
        "freeze_pip_freeze_paths": [str(baseline_pip_freeze), str(no_normalizer_pip_freeze)],
        "freeze_intent_run_count": 1,
        "git_commit_count": 1,
        "git_commits": ["a" * 40],
        "incomplete_live_authority_labels": ["live:core:no_normalizer:1", "live:core:reducer:1"],
        "latest_code_claim_run_count": 1,
        "live_run_count": 3,
        "live_runs_with_provenance": 2,
        "missing_labels": ["live:core:reducer:1"],
        "qualified_live_authority_run_count": 1,
        "run_count_with_provenance": 2,
        "run_logs_total": 3,
        "run_provenance_coverage": 2 / 3,
        "single_git_commit": True,
    }
    assert report["reproducibility"]["artifact_case_count"] == 1
    assert report["reproducibility"]["artifact_dirs_total"] == 1
    assert report["reproducibility"]["artifact_dirs_existing"] == 1
    assert report["reproducibility"]["artifact_dirs_with_core_payload"] == 1
    assert report["reproducibility"]["artifact_dirs_with_reproduce_script"] == 1
    assert report["reproducibility"]["artifact_dirs_with_reduced_reproducer"] == 1
    assert report["reproducibility"]["artifact_dirs_with_standalone_reproducer"] == 1
    assert report["reproducibility"]["artifact_dirs_with_triage_report"] == 1
    assert report["reproducibility"]["artifact_reproducer_coverage"] == 1.0
    assert report["reproducibility"]["issue_bundle"] == {
        "all_selected_reproducers_executed": True,
        "clean_execution": True,
        "compile_failure_count": 0,
        "executed_reproducer_count": 2,
        "executed_reproducer_attempt_count": 4,
        "executed_reproducer_coverage": 1.0,
        "extracted_reproducer_count": 2,
        "families": ["ready_family@polars", "dedup_family@duckdb"],
        "family_count": 2,
        "flaky_reproducer_count": 0,
        "generated_at": "2026-05-28T11:22:01Z",
        "generated_by": "datadiff issue-bundle",
        "issue_count": 2,
        "issue_paths": ["new_issue/ready.md", "new_issue/dedup.md"],
        "missing_reproducer_count": 0,
        "nonzero_exit_count": 0,
        "nonzero_exit_attempt_count": 0,
        "path": str(issue_bundle_dir / "manifest.json"),
        "present": True,
        "reproducer_paths": [
            "new_issue/generated/issue-bundles/reproducers/ready.py",
            "new_issue/generated/issue-bundles/reproducers/dedup.py",
        ],
        "repeat_count": 2,
        "run_reproducers": True,
        "schema_version": "issue-bundle-v1",
        "timeout_count": 0,
        "timeout_attempt_count": 0,
    }
    assert report["bug_discovery"]["candidate_bug_family_count"] == 1
    assert report["bug_discovery"]["candidate_family_first_seen_count"] == 1
    assert report["bug_discovery"]["candidate_family_first_seen"] == {
        "fresh_root@polars": {
            "case_id": f"case-{baseline_run.stem}-1",
            "case_index": 1,
            "case_seed": 1,
            "elapsed_s": 0.2,
            "preset": "baseline",
            "run_file": str(baseline_run),
            "seed": "1",
            "target_suite": "core",
            "variant_label": "baseline",
        }
    }
    assert report["soundness"]["false_positive_count"] == 1
    assert report["closed_loop_feedback"]["raw_new_behavior_cases"] == 3
    assert report["closed_loop_feedback"]["signal_new_behavior_cases"] == 2
    assert report["closed_loop_feedback"]["raw_new_behavior_rate"] == 0.5
    assert report["closed_loop_feedback"]["signal_new_behavior_rate"] == 2 / 6
    assert report["closed_loop_feedback"]["feedback_mutation_cases"] >= 1
    assert report["closed_loop_feedback"]["quality_pass_rate"] > 0.0
    assert report["closed_loop_feedback"]["seed_schedule_delta_per_case"] > 0.0
    assert report["closed_loop_feedback"]["semantic_focus_families"] == ["join_membership"]
    assert report["closed_loop_feedback"]["semantic_focus_signals"] == ["topk_filter_pushdown"]
    assert report["closed_loop_feedback"]["configured_guidance_targets"] == ["groupby", "topk"]
    assert report["closed_loop_feedback"]["configured_effective_guidance_targets"] == [
        "groupby",
        "semantic_family:join_membership",
        "semantic_signal:topk_filter_pushdown",
        "topk",
    ]
    assert report["closed_loop_feedback"]["configured_semantic_focus_families"] == ["join_membership"]
    assert report["closed_loop_feedback"]["configured_semantic_focus_signals"] == ["topk_filter_pushdown"]
    assert report["closed_loop_feedback"]["configured_discovery_biases"] == ["groupby|semantic_signal:|keep"]
    assert report["closed_loop_feedback"]["matched_semantic_target_cases"] == 3
    assert report["closed_loop_feedback"]["discovery_bias_hit_cases"] == 3
    assert report["closed_loop_feedback"]["top_matched_semantic_targets"]["groupby"] == 3
    assert report["closed_loop_feedback"]["top_observed_semantic_families"]["join_membership"] == 6
    assert report["closed_loop_feedback"]["top_observed_semantic_signals"]["topk_filter_pushdown"] == 6
    assert report["closed_loop_feedback"]["feedback_operator_affinity_hit_rate"] >= 0.0
    assert "- Stage backend avg ms:" in md
    assert report["scheduler_effectiveness"]["manifest_count"] == 1
    assert report["scheduler_effectiveness"]["completed_run_count"] == 1
    assert report["scheduler_effectiveness"]["adaptive_enabled"] is True
    assert report["scheduler_effectiveness"]["adaptive_final_arm_count"] == 2
    assert report["scheduler_effectiveness"]["adaptive_avg_reward_signal"] == (3.4 + 0.4) / 2
    assert report["scheduler_effectiveness"]["adaptive_max_reward_signal"] == 3.4
    assert report["scheduler_effectiveness"]["adaptive_avg_mean_reward"] == (4.2 + 1.1) / 2
    assert report["scheduler_effectiveness"]["adaptive_final_arms"][0]["arm_id"] == "core:baseline:seed1"
    assert report["scheduler_effectiveness"]["local_source_scheduler"] == {
        "enabled": True,
        "exploration_weight": 0.2,
    }
    assert report["scheduler_effectiveness"]["lane_yield"][0]["lane_id"] == "arrow_layout"
    assert report["candidate_pipeline"]["manifest_count"] == 1
    assert report["candidate_pipeline"]["candidate_count"] == 2
    assert report["candidate_pipeline"]["reproduced_count"] == 1
    assert report["candidate_pipeline"]["strategy_snapshot_count"] == 1
    assert report["candidate_pipeline"]["strategy_learning_count"] == 1
    assert report["offline_oracle"]["classified_findings"] == 4
    assert report["offline_oracle"]["buckets"] == {
        "new_bug": 1,
        "known_bug": 1,
        "false_positive": 1,
        "semantic_divergence": 1,
        "needs_triage": 0,
        "unclassified": 0,
    }
    assert report["ablation"]["ablation_run_groups"] == 2
    assert report["ablation"]["ablation_variant_ids"] == ["no_normalizer", "reducer"]
    assert report["ablation"]["ablation_modules"] == ["reducer", "semantic_normalizer"]
    assert "reducer" not in report["ablation"]["missing_ablation_modules"]
    assert report["comparisons"][0]["evidence_bytes_per_case_delta"] != 0
    assert report["evidence_chain"]["artifact_dirs"] == [str(artifact_dir)]
    assert report["evidence_chain"]["issue_bundle_manifest"] == str(issue_bundle_dir / "manifest.json")
    assert report["evidence_chain"]["issue_bundle_reproducers"] == [
        "new_issue/generated/issue-bundles/reproducers/ready.py",
        "new_issue/generated/issue-bundles/reproducers/dedup.py",
    ]
    assert "Aggregate JSON" in md
    assert "## Evidence Chain" in md
    assert "## Offline Oracle Buckets" in md
    assert "## Closed-Loop Feedback" in md
    assert "Structured semantic focus families" in md
    assert "Configured guidance targets" in md
    assert "Top observed semantic families" in md
    assert "Top selected feedback operators" in md
    assert "- Matrix ids: 1 (module_ablation)" in md
    assert "- Comparison groups: 1 (module_ablation)" in md
    assert "- Variants: 3 (baseline, no_normalizer, reducer)" in md
    assert "- Raw new behavior cases: 3 (50.0%)" in md
    assert "- Signal new behavior cases: 2 (33.3%)" in md
    assert "## Scheduler Effectiveness" in md
    assert "- Adaptive final arms: 2" in md
    assert "- Adaptive avg reward signal: 1.90" in md
    assert "- Adaptive max reward signal: 3.40" in md
    assert "- Local source scheduler: {'enabled': True, 'exploration_weight': 0.2}" in md
    assert "### Adaptive Final Arms" in md
    assert "| core:baseline:seed1 | core | 3 | 4.20 | 3.40 | 3.70 | 0 |" in md
    assert "## Candidate Pipeline" in md
    assert "### Candidate Family First Seen" in md
    assert "fresh_root@polars" in md
    assert "- Known bug: 1" in md
    assert "Run logs scanned: 3/3" in md
    assert "Run provenance recorded: 2/3" in md
    assert "Fully qualified latest-live authority runs: 1" in md
    assert "All live runs satisfy authority/freeze gates: false" in md
    assert "Artifact reproducer coverage: 100.0%" in md
    assert "Run-provenance freeze manifests indexed: 2" in md
    assert "Run-provenance pip-freeze artifacts indexed: 2" in md
    assert "Run-provenance git-status artifacts indexed: 2" in md
    assert "Run-provenance git-diff artifacts indexed: 1" in md
    assert "Run-provenance launcher-env artifacts indexed: 2" in md
    assert "Run-provenance strategy-snapshot artifacts indexed: 2" in md
    assert "Run-provenance strategy-learning artifacts indexed: 2" in md
    assert "Run-provenance freeze manifests indexed: 2" in md
    assert "Issue bundle reproducers executed: 2/2" in md
    assert "Issue bundle reproducer attempts: 4" in md
    assert "Issue bundle flaky reproducers: 0" in md
    assert "Issue bundle clean execution: true" in md
    assert "Ablated modules covered: reducer, semantic_normalizer" in md
    assert "Ablation variants: no_normalizer, reducer" in md
    assert "Reference groups: 1" in md
    assert report["ablation"]["reference_run_groups"] == 1
    assert report["comparisons"][0]["reference_variant_id"] == "baseline"
    assert report["comparisons"][0]["reference_variant_label"] == "baseline"
    assert report["comparisons"][0]["variant_label"] in {"no_normalizer", "reducer"}
    assert "evidence bytes/case ratio" in md

    def fail_iter_jsonl(path):
        raise AssertionError(f"summary-only report unexpectedly scanned {path}")
        yield  # pragma: no cover

    monkeypatch.setattr(methodology_report, "_iter_jsonl", fail_iter_jsonl)

    _, summary_only_json = write_methodology_report(manifest, scan_run_logs=False)

    summary_only_report = json.loads(summary_only_json.read_text(encoding="utf-8"))
    assert summary_only_report["run_log_scan"] == {
        "run_logs_missing": 0,
        "run_logs_scan_skipped": True,
        "run_logs_scanned": 0,
        "run_logs_total": 3,
    }
    assert summary_only_report["offline_oracle"]["classified_findings"] == 0
    assert summary_only_report["reproducibility"]["artifact_dirs_total"] == 0
    assert summary_only_report["reproducibility"]["run_provenance"]["qualified_live_authority_run_count"] == 1
    assert summary_only_report["reproducibility"]["issue_bundle"]["clean_execution"] is True


def test_methodology_report_uses_component_focus_for_ablation_classification(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "PROJECT_ROOT", tmp_path)

    baseline_run = runs_dir / "run-base.jsonl.gz"
    contrast_run = runs_dir / "run-contrast.jsonl.gz"
    _write_run(baseline_run, findings_by_index={}, artifact_dir=None)
    _write_run(contrast_run, findings_by_index={}, artifact_dir=None)
    manifest = runs_dir / "experiment-methodology-structured.json"
    dump_json(
        {
            "presets": ["stable_base", "focus_variant"],
            "seeds": [1],
            "target_suite": "core",
            "target_suites": ["core"],
            "evidence_mode": "live",
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "stable_base",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "stable_base",
                    "variant_title": "stable_base",
                    "base_preset": "stable_base",
                    "comparison_role": "baseline",
                    "component_focus": "",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2"],
                    "analysis_tags": ["ablation", "baseline"],
                    "seed": 1,
                    "evidence_mode": "live",
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "focus_variant",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "focus_variant",
                    "variant_title": "focus_variant",
                    "base_preset": "stable_base",
                    "comparison_role": "contrast",
                    "component_focus": "semantic_normalizer",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2"],
                    "analysis_tags": ["ablation", "noise_control"],
                    "seed": 1,
                    "evidence_mode": "live",
                    "run_file": str(contrast_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )

    _, json_path = write_methodology_report(manifest, scan_run_logs=False)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["ablation"]["ablation_variant_ids"] == ["focus_variant"]
    assert report["ablation"]["ablation_modules"] == ["semantic_normalizer"]
    assert report["ablation"]["missing_ablation_modules"] == []


def test_methodology_report_prefers_structured_contrast_rows_in_comparisons(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "PROJECT_ROOT", tmp_path)

    baseline_run = runs_dir / "run-base-contrast-only.jsonl.gz"
    contrast_run = runs_dir / "run-contrast-contrast-only.jsonl.gz"
    auxiliary_run = runs_dir / "run-aux-contrast-only.jsonl.gz"
    _write_run(baseline_run, findings_by_index={}, artifact_dir=None)
    _write_run(contrast_run, findings_by_index={}, artifact_dir=None)
    _write_run(auxiliary_run, findings_by_index={}, artifact_dir=None)
    manifest = runs_dir / "experiment-methodology-contrast-only.json"
    dump_json(
        {
            "presets": ["stable_base", "focus_variant", "workflow_probe"],
            "seeds": [1],
            "target_suite": "core",
            "target_suites": ["core"],
            "evidence_mode": "ablation",
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "stable_base",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "stable_base",
                    "variant_title": "stable_base",
                    "base_preset": "stable_base",
                    "comparison_role": "baseline",
                    "component_focus": "",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2"],
                    "analysis_tags": ["ablation", "baseline"],
                    "seed": 1,
                    "evidence_mode": "ablation",
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "focus_variant",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "focus_variant",
                    "variant_title": "focus_variant",
                    "base_preset": "stable_base",
                    "comparison_role": "contrast",
                    "component_focus": "semantic_normalizer",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2"],
                    "analysis_tags": ["ablation", "noise_control"],
                    "seed": 1,
                    "evidence_mode": "ablation",
                    "run_file": str(contrast_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "workflow_probe",
                    "matrix_id": "module_ablation",
                    "comparison_group": "module_ablation",
                    "variant_id": "workflow_probe",
                    "variant_title": "workflow_probe",
                    "base_preset": "stable_base",
                    "comparison_role": "",
                    "component_focus": "",
                    "scope_kind": "core",
                    "oracle_profile": "differential",
                    "rq_tags": ["RQ2"],
                    "analysis_tags": ["ablation", "workflow"],
                    "seed": 1,
                    "evidence_mode": "ablation",
                    "run_file": str(auxiliary_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )

    _, json_path = write_methodology_report(manifest, scan_run_logs=False)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert [row["preset"] for row in report["comparisons"]] == ["focus_variant"]


def test_methodology_report_prefers_structured_aggregate_json_variant_rows(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    manifest = tmp_path / "runs" / "experiment-methodology-json-preferred.json"
    manifest.parent.mkdir(parents=True)
    dump_json({"runs": []}, manifest)
    monkeypatch.setattr(methodology_report, "REPORTS_DIR", reports_dir)

    def fake_write_experiment_summary(manifest_file, *, refresh=False):
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / f"experiment-summary-{manifest_file.stem}.md"
        csv_path = reports_dir / f"experiment-summary-{manifest_file.stem}.csv"
        aggregate_csv_path = reports_dir / f"{md_path.stem}-aggregates.csv"
        aggregate_json_path = reports_dir / f"{md_path.stem}-aggregates.json"
        md_path.write_text("# Summary\n", encoding="utf-8")
        csv_path.write_text(
            "target_suite,preset,matrix_id,comparison_group,variant_id,scope_kind,oracle_profile,cases,findings,candidate_bug_cases,evidence_mode,run_file\n",
            encoding="utf-8",
        )
        aggregate_csv_path.write_text(
            "target_suite,preset,matrix_id,comparison_group,variant_id,variant_title,scope_kind,oracle_profile,analysis_tags,candidate_bug_case_rate,cases,candidate_bug_cases\n"
            "csv_suite,csv_baseline,csv_matrix,csv_group,csv_baseline,csv_baseline,csv_scope,csv_oracle,csv_tag,0.0,1,0\n",
            encoding="utf-8",
        )
        aggregate_json_path.write_text(
            json.dumps(
                {
                    "variant_rows": [
                        {
                            "target_suite": "json_suite",
                            "preset": "baseline",
                            "matrix_id": "json_matrix",
                            "comparison_group": "json_group",
                            "variant_id": "baseline",
                            "variant_title": "baseline",
                            "variant_label": "baseline",
                            "scope_kind": "json_scope",
                            "oracle_profile": "json_oracle",
                            "analysis_tags": "json_tag",
                            "rq_tags": "RQ2",
                            "candidate_bug_case_rate": 0.0,
                            "cases": 2,
                            "candidate_bug_cases": 0,
                            "findings": 0,
                            "false_positive_count": 0,
                            "semantic_focus_families": "join_membership",
                            "semantic_focus_signals": "topk_filter_pushdown",
                            "evidence_mode": "live",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return md_path, csv_path

    monkeypatch.setattr(methodology_report, "write_experiment_summary", fake_write_experiment_summary)

    _, json_path = write_methodology_report(manifest, scan_run_logs=False)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert report["coverage"]["target_suites"] == ["json_suite"]
    assert report["coverage"]["matrix_ids"] == ["json_matrix"]
    assert report["coverage"]["scope_kinds"] == ["json_scope"]
    assert report["coverage"]["oracle_profiles"] == ["json_oracle"]


def test_methodology_report_normalizes_manifest_experiment_meta_before_ablation_expectations(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(reporter, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(methodology_report, "PROJECT_ROOT", tmp_path)

    baseline_run = runs_dir / "run-base-normalized.jsonl.gz"
    focus_run = runs_dir / "run-focus-normalized.jsonl.gz"
    _write_run(baseline_run, findings_by_index={}, artifact_dir=None)
    _write_run(focus_run, findings_by_index={}, artifact_dir=None)
    manifest = runs_dir / "experiment-methodology-normalized-meta.json"
    dump_json(
        {
            "presets": ["baseline", "no_normalizer"],
            "seeds": [1],
            "target_suite": "core",
            "target_suites": ["core"],
            "evidence_mode": "ablation",
            "experiment_meta": {
                "matrix_id": "module_ablation",
                "comparison_group": "module_ablation",
                "analysis_tags": ["ablation", ""],
                "variant_by_preset": {
                    "baseline": {
                        "variant_id": "baseline",
                        "comparison_role": "baseline",
                        "analysis_tags": ["ablation", ""],
                    },
                    "no_normalizer": {
                        "variant_id": "no_normalizer",
                        "comparison_role": "contrast",
                        "component_focus": "semantic_normalizer",
                        "analysis_tags": ["ablation", "noise_control", ""],
                    },
                },
            },
            "runs": [
                {
                    "target_suite": "core",
                    "preset": "baseline",
                    "seed": 1,
                    "evidence_mode": "ablation",
                    "run_file": str(baseline_run),
                    "report": "",
                },
                {
                    "target_suite": "core",
                    "preset": "no_normalizer",
                    "seed": 1,
                    "evidence_mode": "ablation",
                    "run_file": str(focus_run),
                    "report": "",
                },
            ],
        },
        manifest,
    )

    _, json_path = write_methodology_report(manifest, scan_run_logs=False)
    report = json.loads(json_path.read_text(encoding="utf-8"))

    assert report["ablation"]["ablation_variant_ids"] == ["no_normalizer"]
    assert report["ablation"]["ablation_modules"] == ["semantic_normalizer"]
    assert "semantic_normalizer" not in report["ablation"]["missing_ablation_modules"]
