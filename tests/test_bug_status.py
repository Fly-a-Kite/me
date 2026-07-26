import json

from datadiff import bug_status, cli
from datadiff.bug_status import (
    build_bug_status,
    build_issue_status,
    render_bug_status_markdown,
    render_issue_status_markdown,
    write_bug_status_outputs,
    write_issue_status_outputs,
)
from datadiff.cli import build_parser


def test_build_bug_status_summarizes_lightweight_evidence(tmp_path):
    latest = tmp_path / "latest_confirmations.json"
    latest.write_text(
        json.dumps(
            {
                "confirmations": [
                    {
                        "family": "grouped_topk_null_sort_key@datafusion",
                        "issue_url": "https://github.com/apache/datafusion/issues/1",
                        "discovery_credit": "datadiff_submitted",
                        "upstream_status": "upstream_labeled_bug",
                    },
                    {
                        "root_cause": "vector_division_rounding",
                        "suspicious_backends": ["polars"],
                        "issue_url": "https://github.com/pola-rs/polars/issues/2",
                        "discovery_credit": "datadiff_submitted",
                        "upstream_status": "fixed_upstream",
                    },
                    {
                        "family": "not_confirmed@duckdb",
                        "issue_url": "https://github.com/duckdb/duckdb/issues/3",
                        "discovery_credit": "datadiff_submitted",
                        "upstream_status": "needs_triage",
                    },
                    {
                        "family": "similar_existing@pandas",
                        "issue_url": "https://github.com/pandas-dev/pandas/issues/4",
                        "discovery_credit": "similar_existing",
                        "upstream_status": "upstream_labeled_bug",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    new_issue = tmp_path / "new_issue"
    generated = new_issue / "generated"
    old_issue = tmp_path / "old_issue"
    generated.mkdir(parents=True)
    old_issue.mkdir()
    (new_issue / "pending.md").write_text(
        "# Pending Issue\n\n| Item | Value |\n| --- | --- |\n| Current status | Needs stable reproducer; repeated issue-bundle attempts are flaky |\n",
        encoding="utf-8",
    )
    (new_issue / "submitted.md").write_text(
        "# Submitted Issue\n\n| 项目 | 内容 |\n| --- | --- |\n| 当前状态 | 上游 issue closed，标签含 `bug`、`accepted` |\n",
        encoding="utf-8",
    )
    (generated / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bug-audit-v1",
                "generated_by": "datadiff bug-audit",
                "candidate_bug_families": ["audit_family@pyarrow"],
                "issue_files": ["new_issue/generated/audit_family.md"],
            }
        ),
        encoding="utf-8",
    )
    (generated / "hunt-fresh-candidates.json").write_text(
        json.dumps(
            {
                "source_run_file": "runs/run.jsonl.gz",
                "fresh_candidate_bug_families": {
                    "fresh_family@polars": 2,
                    "polars_vector_division_rounding@polars": 4,
                },
                "candidate_row_count": 2,
            }
        ),
        encoding="utf-8",
    )
    (generated / "stale-label-fresh-candidates.json").write_text(
        json.dumps(
            {
                "source_run_file": "runs/stale.jsonl.gz",
                "fresh_candidate_bug_families": {
                    "join_semantics@datafusion": 1,
                    "groupby_aggregation@pyarrow": 1,
                    "metamorphic_limit_idempotence@datafusion": 1,
                    "string_expression@duckdb": 1,
                },
                "candidate_row_count": 4,
                "candidate_rows": [
                    {
                        "case": {
                            "case_id": "case-stale-distinct",
                            "seed": 7,
                            "tables": [
                                {
                                    "name": "t0",
                                    "columns": [{"name": "s", "type": "str", "nullable": True}],
                                    "rows": [{"s": None}, {"s": ""}],
                                }
                            ],
                            "program": {
                                "program_id": "prog-stale-distinct",
                                "seed": 7,
                                "operations": [
                                    {"op": "distinct", "columns": ["s"]},
                                    {"op": "sort", "keys": [{"column": "s", "ascending": True, "nulls": "first"}]},
                                    {"op": "limit", "n": 1},
                                ],
                            },
                        },
                        "normalized": {
                            "duckdb": {"backend": "duckdb", "status": "ok", "columns": ["s"], "rows": [[None]]},
                            "datafusion": {"backend": "datafusion", "status": "ok", "columns": ["s"], "rows": [[""]]},
                        },
                        "findings": [
                                {
                                    "kind": "semantic_output_mismatch",
                                    "root_cause": "join_semantics",
                                    "suspicious_backends": ["datafusion"],
                                    "triage_verdict": "candidate_implementation_bug",
                                }
                            ],
                        },
                    {
                        "case": {
                            "case_id": "case-stale-limit-idempotence",
                            "seed": 8,
                            "tables": [
                                {
                                    "name": "t0",
                                    "columns": [{"name": "x", "type": "int", "nullable": True}],
                                    "rows": [{"x": None}, {"x": 1}, {"x": 2}],
                                }
                            ],
                            "program": {
                                "program_id": "prog-stale-limit-idempotence",
                                "seed": 8,
                                "operations": [
                                    {"op": "sort", "keys": [{"column": "x", "ascending": True, "nulls": "first"}]},
                                    {"op": "limit", "n": 2},
                                ],
                            },
                        },
                        "normalized": {
                            "datafusion": {"backend": "datafusion", "status": "ok", "columns": ["x"], "rows": [[1]]}
                        },
                        "findings": [
                            {
                                "kind": "metamorphic_limit_idempotence_violation",
                                "oracle": "metamorphic",
                                "root_cause": "metamorphic_limit_idempotence",
                                "suspicious_backends": ["datafusion"],
                            }
                        ],
                    },
                    {
                        "case": {
                            "case_id": "case-stale-pyarrow-empty-bool-aggregate",
                            "seed": 10,
                            "tables": [
                                {
                                    "name": "t0",
                                    "columns": [
                                        {"name": "id", "type": "int", "nullable": False},
                                        {"name": "flag", "type": "bool", "nullable": True},
                                    ],
                                    "rows": [{"id": 0, "flag": None}, {"id": 1, "flag": True}],
                                }
                            ],
                            "program": {
                                "program_id": "prog-stale-pyarrow-empty-bool-aggregate",
                                "seed": 10,
                                "operations": [
                                    {"op": "filter", "column": "id", "cmp": "<", "value": 0},
                                    {
                                        "op": "aggregate",
                                        "aggs": [
                                            {"column": "flag", "func": "any", "as": "any_flag"},
                                            {"column": "flag", "func": "all", "as": "all_flag"},
                                        ],
                                    },
                                    {
                                        "op": "filter",
                                        "column": "any_flag",
                                        "cmp": "bool_is_not_true",
                                        "value": None,
                                    },
                                ],
                            },
                        },
                        "raw_results": {
                            "pyarrow": {
                                "status": "error",
                                "error_type": "ArrowInvalid",
                                "error": "Invalid null value",
                            }
                        },
                        "normalized": {
                            "pandas": {
                                "backend": "pandas",
                                "status": "ok",
                                "columns": ["all_flag", "any_flag"],
                                "rows": [[None, None]],
                            },
                            "pyarrow": {
                                "backend": "pyarrow",
                                "status": "error",
                                "columns": [],
                                "rows": [],
                                "error_type": "ArrowInvalid",
                                "error": "Invalid null value",
                            },
                        },
                        "findings": [
                            {
                                "kind": "accept_reject_mismatch",
                                "root_cause": "groupby_aggregation",
                                "suspicious_backends": ["pyarrow"],
                                "triage_verdict": "candidate_implementation_bug",
                            }
                        ],
                    },
                    {
                        "case": {
                            "case_id": "case-stale-sort-limit-tie",
                            "seed": 9,
                            "tables": [
                                {
                                    "name": "t0",
                                    "columns": [
                                        {"name": "id", "type": "int", "nullable": False},
                                        {"name": "g", "type": "str", "nullable": True},
                                        {"name": "s", "type": "str", "nullable": True},
                                    ],
                                    "rows": [
                                        {"id": 1, "g": "A", "s": "x"},
                                        {"id": 2, "g": "", "s": ""},
                                        {"id": 2, "g": "", "s": "Beta"},
                                        {"id": 3, "g": "", "s": "Gamma"},
                                    ],
                                }
                            ],
                            "program": {
                                "program_id": "prog-stale-sort-limit-tie",
                                "seed": 9,
                                "operations": [
                                    {
                                        "op": "mutate",
                                        "column": "g_token",
                                        "expr": {
                                            "kind": "string_replace",
                                            "source": "g",
                                            "old": " ",
                                            "new": "_",
                                        },
                                    },
                                    {
                                        "op": "sort",
                                        "keys": [
                                            {"column": "g_token", "ascending": True, "nulls": "last"},
                                            {"column": "id", "ascending": True, "nulls": "last"},
                                        ],
                                    },
                                    {"op": "select", "columns": ["id", "g_token", "s"]},
                                    {"op": "limit", "n": 1},
                                ],
                            },
                        },
                        "normalized": {
                            "duckdb": {
                                "backend": "duckdb",
                                "status": "ok",
                                "columns": ["id", "g_token", "s"],
                                "rows": [[2, "", ""]],
                            },
                            "pandas": {
                                "backend": "pandas",
                                "status": "ok",
                                "columns": ["id", "g_token", "s"],
                                "rows": [[2, "", "Beta"]],
                            },
                        },
                        "findings": [
                            {
                                "kind": "semantic_output_mismatch",
                                "root_cause": "string_expression",
                                "suspicious_backends": ["duckdb"],
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (generated / "discovery-run-manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-26T00:00:00Z",
                "target_suite": "latest_all_engines",
                "preset": "live_deep_organic",
                "classification": {
                    "fresh_candidate_bug_families": {"fresh_family@polars": 2},
                    "known_saturated_candidate_bug_families": {"known_family@datafusion": 1},
                },
            }
        ),
        encoding="utf-8",
    )
    (generated / "discovery-campaign-manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-05-26T01:00:00Z",
                "lane_ids": ["arrow_layout"],
                "seeds": [11],
                "cases_per_lane_seed": 3,
                "summary": {
                    "run_count": 1,
                    "fresh_candidate_bug_families": {"fresh_family@pyarrow": 1},
                    "known_saturated_candidate_bug_families": {},
                },
            }
        ),
        encoding="utf-8",
    )
    bundle = generated / "issue-bundles"
    bundle.mkdir()
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "issue-bundle-v1",
                "generated_at": "2026-05-26T02:00:00Z",
                "generated_by": "datadiff issue-bundle",
                "manifest_path": "new_issue/generated/issue-bundles/manifest.json",
                "markdown_path": "new_issue/generated/issue-bundles/manifest.md",
                "summary": {
                    "family_count": 2,
                    "families": ["bundle_family@polars", "bundle_family@pyarrow"],
                    "extracted_reproducer_count": 3,
                    "missing_reproducer_count": 0,
                    "compile_failure_count": 0,
                    "executed_reproducer_count": 3,
                    "executed_reproducer_attempt_count": 6,
                    "flaky_reproducer_count": 1,
                    "nonzero_exit_count": 1,
                    "nonzero_exit_attempt_count": 1,
                    "timeout_count": 0,
                    "timeout_attempt_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (old_issue / "README.md").write_text(
        "https://github.com/apache/arrow/issues/32171\n"
        "https://github.com/apache/arrow/issues/32171\n"
        "https://github.com/duckdb/duckdb/issues/22418\n",
        encoding="utf-8",
    )

    status = build_bug_status(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
    )
    assert status == build_issue_status(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
    )

    summary = status["summary"]
    assert summary["confirmed_latest_count"] == 2
    assert summary["confirmed_latest_families"] == [
        "grouped_topk_null_sort_key@datafusion",
        "vector_division_rounding@polars",
    ]
    assert summary["manual_new_issue_draft_count"] == 2
    assert summary["pending_manual_issue_draft_count"] == 1
    assert summary["audit_candidate_families"] == ["audit_family@pyarrow"]
    assert summary["fresh_candidate_families"] == {"fresh_family@polars": 2}
    assert summary["recorded_fresh_candidate_families"] == {
        "datafusion_limit_idempotence@datafusion": 1,
        "distinct_null_topk@datafusion": 1,
        "fresh_family@polars": 2,
        "polars_vector_division_rounding@polars": 4,
    }
    stale_evidence = next(
        item for item in status["fresh_candidate_evidence"] if item["path"].endswith("stale-label-fresh-candidates.json")
    )
    assert stale_evidence["raw_fresh_candidate_bug_families"] == {
        "groupby_aggregation@pyarrow": 1,
        "join_semantics@datafusion": 1,
        "metamorphic_limit_idempotence@datafusion": 1,
        "string_expression@duckdb": 1,
    }
    assert stale_evidence["fresh_candidate_bug_families"] == {
        "datafusion_limit_idempotence@datafusion": 1,
        "distinct_null_topk@datafusion": 1,
    }
    assert summary["discovery_run_manifest_count"] == 1
    assert summary["discovery_campaign_manifest_count"] == 1
    assert summary["discovery_run_manifest_count"] == 1
    assert summary["discovery_campaign_manifest_count"] == 1
    assert summary["discovery_workflow_manifest_count"] == 2
    assert summary["issue_bundle_present"] is True
    assert summary["issue_bundle_family_count"] == 2
    assert summary["issue_bundle_reproducer_count"] == 3
    assert summary["issue_bundle_missing_reproducer_count"] == 0
    assert summary["issue_bundle_compile_failure_count"] == 0
    assert summary["issue_bundle_executed_reproducer_count"] == 3
    assert summary["issue_bundle_executed_reproducer_attempt_count"] == 6
    assert summary["issue_bundle_flaky_reproducer_count"] == 1
    assert summary["issue_bundle_nonzero_exit_count"] == 1
    assert summary["issue_bundle_nonzero_exit_attempt_count"] == 1
    assert summary["issue_bundle_timeout_count"] == 0
    assert summary["issue_bundle_timeout_attempt_count"] == 0
    assert status["issue_bundle_manifest"]["families"] == ["bundle_family@polars", "bundle_family@pyarrow"]
    assert status["discovery_campaign_manifests"][0]["lane_ids"] == ["arrow_layout"]
    assert status["discovery_campaign_manifests"][0]["lane_ids"] == ["arrow_layout"]
    assert summary["old_known_upstream_issue_count"] == 2


def test_write_bug_status_outputs_writes_json_and_markdown(tmp_path):
    status = {
        "schema_version": "bug-status-v1",
        "generated_at": "2026-05-26T12:00:00Z",
        "summary": {
            "confirmed_latest_count": 1,
            "confirmed_latest_families": ["confirmed@engine"],
            "audit_candidate_family_count": 1,
            "audit_candidate_families": ["audit@engine"],
            "fresh_candidate_family_count": 0,
            "fresh_candidate_families": {},
            "recorded_fresh_candidate_family_count": 0,
            "recorded_fresh_candidate_families": {},
            "pending_manual_issue_draft_count": 0,
            "pending_manual_issue_drafts": [],
            "issue_bundle_nonzero_exit_count": 0,
            "issue_bundle_flaky_reproducer_count": 0,
            "issue_bundle_timeout_count": 0,
            "old_known_upstream_issue_count": 3,
        },
    }

    json_path, md_path = write_bug_status_outputs(status, output_dir=tmp_path)

    assert json_path.name == "bug-status-20260526T120000.json"
    assert md_path.name == "bug-status-20260526T120000.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "bug-status-v1"
    assert write_issue_status_outputs(status, output_dir=tmp_path) == (json_path, md_path)
    markdown = md_path.read_text(encoding="utf-8")
    assert "`confirmed@engine`" in markdown
    assert "Discovery workflow manifests" in markdown
    assert "Issue bundle nonzero exits" in markdown
    assert "DataDiffFuzz Bug Status" in render_bug_status_markdown(status)
    assert render_issue_status_markdown(status) == render_bug_status_markdown(status)


def test_build_bug_status_can_skip_historical_workflow_payloads(tmp_path, monkeypatch):
    new_issue = tmp_path / "new_issue"
    generated = new_issue / "generated"
    old_issue = tmp_path / "old_issue"
    generated.mkdir(parents=True)
    old_issue.mkdir()

    def fail_if_scanned(_directory):
        raise AssertionError("historical workflow evidence must not be scanned")

    monkeypatch.setattr(bug_status, "_collect_fresh_candidate_evidence", fail_if_scanned)
    monkeypatch.setattr(bug_status, "_collect_discovery_run_manifests", fail_if_scanned)
    monkeypatch.setattr(bug_status, "_collect_discovery_campaign_manifests", fail_if_scanned)

    status = build_issue_status(
        latest_confirmation_files=[],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        scan_generated_workflow_evidence=False,
    )

    assert status["inputs"]["scan_generated_workflow_evidence"] is False
    assert status["fresh_candidate_evidence"] == []
    assert status["discovery_run_manifests"] == []
    assert status["discovery_campaign_manifests"] == []
    assert status["summary"]["discovery_workflow_manifest_count"] == 0


def test_cli_parses_bug_status_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bug-status",
            "--json",
            "--latest-confirmations",
            "experiments/latest_confirmations.json,other.json",
            "--new-issue-dir",
            "new_issue",
            "--old-issue-dir",
            "old_issue",
            "--generated-issue-dir",
            "new_issue/generated",
            "--write-report",
            "--output-dir",
            "reports",
        ]
    )

    assert args.cmd == "bug-status"
    assert args.json is True
    assert args.latest_confirmations == "experiments/latest_confirmations.json,other.json"
    assert args.write_report is True


def test_cli_bug_status_prints_json(monkeypatch, capsys):
    status = {
        "schema_version": "bug-status-v1",
        "summary": {
            "confirmed_latest_count": 1,
            "confirmed_latest_families": ["confirmed@engine"],
            "audit_candidate_family_count": 0,
            "audit_candidate_families": [],
            "fresh_candidate_family_count": 0,
            "fresh_candidate_families": {},
            "recorded_fresh_candidate_family_count": 0,
            "recorded_fresh_candidate_families": {},
            "pending_manual_issue_draft_count": 0,
            "pending_manual_issue_drafts": [],
            "old_known_upstream_issue_count": 0,
        },
    }
    monkeypatch.setattr(cli, "build_bug_status", lambda **kwargs: status)

    args = build_parser().parse_args(["bug-status", "--json"])

    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "bug-status-v1"
