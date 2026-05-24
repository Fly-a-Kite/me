import json
from pathlib import Path

from datadiff.run_journal import build_run_journal_entry, record_run_journal
from datadiff.util import append_jsonl, read_jsonl, run_meta_path


def test_build_run_journal_entry_records_paper_facing_summary(tmp_path):
    run_file = tmp_path / "run-x.jsonl"
    append_jsonl(
        {
            "case": {"case_id": "case-1", "seed": 1},
            "case_index": 0,
            "elapsed_s": 1.25,
            "is_new_behavior": True,
            "findings": [
                {
                    "root_cause": "grouped_topk_null_sort_key",
                    "triage_verdict": "candidate_implementation_bug",
                    "suspicious_backends": ["datafusion"],
                    "false_positive": False,
                }
            ],
        },
        run_file,
    )
    run_meta_path(run_file).write_text(
        json.dumps(
            {
                "executed_cases": 1,
                "elapsed_s": 1.3,
                "throughput_cases_s": 0.77,
                "seed": 1,
                "backends": ["pandas", "duckdb", "datafusion"],
                "targets": [{"family": "sql", "layer": "query_engine"}],
                "common_capabilities": ["sort", "limit"],
                "config": {
                    "generator_profile": "null_agg_topk",
                    "enable_replay_bug": False,
                    "replay_bug_source_issues": [
                        "https://github.com/apache/datafusion/issues/22190",
                        "https://github.com/duckdb/duckdb/issues/22075",
                    ],
                    "guidance_strategy": "guided",
                    "guidance_candidate_pool": 8,
                    "guidance_targets": ["aggregation", "sort_limit"],
                },
                "replay_bug_filter": {
                    "enabled": True,
                    "filtered_candidates": 4,
                    "fallback_candidates": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    entry = build_run_journal_entry(
        run_file,
        {
            "command": "experiment",
            "theme": "paper run",
            "evidence_mode": "live",
            "target_suite": "datafusion_cross",
            "preset": "live_datafusion",
        },
    )

    assert entry["theme"] == "paper run"
    assert entry["evidence_mode"] == "live"
    assert entry["executed_cases"] == 1
    assert entry["common_capabilities_count"] == 2
    assert entry["result_summary"]["candidate_bug_family_count"] == 1
    assert entry["result_summary"]["candidate_bug_families"] == {
        "grouped_topk_null_sort_key@datafusion": 1
    }
    assert entry["result_summary"]["first_candidate_bug_elapsed_s"] == 1.25
    assert entry["config_summary"]["enable_replay_bug"] is False
    assert entry["replay_bug_policy"] == {
        "enable_replay_bug": False,
        "source_issue_count": 2,
        "filter_enabled": True,
        "filtered_candidates": 4,
        "fallback_candidates": 1,
    }


def test_record_run_journal_appends_jsonl_and_markdown(tmp_path):
    run_file = tmp_path / "run-y.jsonl"
    journal_file = tmp_path / "paper-run-journal.jsonl"
    append_jsonl({"case": {"case_id": "case-2", "seed": 2}, "findings": []}, run_file)

    jsonl_path, md_path = record_run_journal(
        run_file,
        context={"theme": "empty live run", "evidence_mode": "live"},
        journal_file=journal_file,
    )

    assert jsonl_path == journal_file
    assert md_path == journal_file.with_suffix(".md")
    rows = read_jsonl(journal_file)
    assert rows[0]["theme"] == "empty live run"
    assert rows[0]["result_summary"]["raw_findings"] == 0
    md = Path(md_path).read_text(encoding="utf-8")
    assert "empty live run" in md
    assert "Replay policy" in md
