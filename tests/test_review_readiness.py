import json

from datadiff import cli
from datadiff.cli import build_parser
from datadiff.review_readiness import (
    ReviewThresholds,
    build_review_readiness,
    render_review_readiness_markdown,
    write_review_readiness_outputs,
)


def _write_minimal_review_project(root):
    for path in [
        "docs",
        "scripts",
        "src/datadiff",
        "tests",
        "new_issue/generated",
        "old_issue",
        "experiments",
    ]:
        (root / path).mkdir(parents=True, exist_ok=True)
    (root / ".gitignore").write_text("runs/\nbugs/\nreports/\ncorpus/\nstudy/\n", encoding="utf-8")
    (root / "README.md").write_text(
        "\n".join(
            [
                "datadiff targets",
                "datadiff bug-audit",
                "datadiff bug-hunt",
                "datadiff bug-sprint",
                "datadiff classify-run",
                "datadiff bug-status",
                "datadiff issue-bundle",
                "datadiff issue-readiness",
                "datadiff methodology-report",
                "datadiff final-readiness",
                "datadiff review-readiness",
                "scripts/run_final_experiments.py --track validation",
            ]
        ),
        encoding="utf-8",
    )
    (root / "docs/project_architecture.md").write_text("architecture", encoding="utf-8")
    (root / "docs/automated_bug_detection.md").write_text("automation", encoding="utf-8")
    (root / "experiments/final_protocol.md").write_text(
        "\n".join(
            [
                "scripts/run_final_experiments.py --track validation",
                "scripts/run_final_experiments.py --track live",
                "scripts/run_final_experiments.py --track historical",
                "scripts/run_final_experiments.py --track seeded",
                "scripts/run_final_experiments.py --track ablation",
                "scripts/run_final_experiments.py --track comparison",
            ]
        ),
        encoding="utf-8",
    )
    (root / "scripts/run_final_experiments.py").write_text(
        """
TRACKS = ("validation", "live", "historical", "seeded", "ablation", "comparison")
paper_run_journal = True

def short_validation_command(args):
    return ["--evidence-mode", "validation"]

def module_ablation_command(args):
    return []

def method_comparison_command(args):
    return []
""".lstrip(),
        encoding="utf-8",
    )
    for path in [
        "src/datadiff/targets.py",
        "src/datadiff/datagen.py",
        "src/datadiff/runner.py",
        "src/datadiff/oracle.py",
        "src/datadiff/metamorphic.py",
        "src/datadiff/triage.py",
        "src/datadiff/bug_audit.py",
        "src/datadiff/bug_status.py",
        "src/datadiff/issue_bundle.py",
        "src/datadiff/issue_readiness.py",
        "src/datadiff/methodology_report.py",
        "src/datadiff/run_journal.py",
        "src/datadiff/final_readiness.py",
    ]:
        (root / path).write_text("# stub\n", encoding="utf-8")
    (root / "src/datadiff/final_readiness.py").write_text(
        "\n".join(
            [
                "paper_run_journal = True",
                "short_validation = True",
                "validation_runs = []",
                "require_validation = True",
                "ablation_runs = []",
                "comparison_runs = []",
                "require_ablation = True",
                "require_comparison = True",
            ]
        ),
        encoding="utf-8",
    )
    for path in [
        "tests/test_methodology_contracts.py",
        "tests/test_targets.py",
        "tests/test_bug_audit.py",
        "tests/test_bug_status.py",
        "tests/test_issue_bundle.py",
        "tests/test_issue_readiness.py",
        "tests/test_final_readiness.py",
        "tests/test_methodology_report.py",
        "tests/test_run_journal.py",
        "tests/test_final_experiment_plan.py",
        "tests/test_reward.py",
    ]:
        (root / path).write_text("# stub\n", encoding="utf-8")
    (root / "tests/test_final_experiment_plan.py").write_text(
        "\n".join(
            [
                "def test_validation_command_gates_short_before_long_runs(): pass",
                "def test_ablation_and_comparison_commands_cover_method_rqs(): pass",
                "def test_final_plan_keeps_paper_run_journal_enabled(): pass",
            ]
        ),
        encoding="utf-8",
    )
    (root / "experiments/latest_confirmations.json").write_text(
        json.dumps(
            {
                "confirmations": [
                    {
                        "family": "confirmed_family@engine",
                        "issue_url": "https://github.com/example/project/issues/1",
                        "upstream_status": "upstream_labeled_bug",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "new_issue/pending.md").write_text(
        "# Pending\n\n| Item | Value |\n| --- | --- |\n| Current status | New candidate issue, not yet submitted upstream |\n",
        encoding="utf-8",
    )
    (root / "new_issue/generated/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bug-audit-v1",
                "candidate_bug_families": ["audit_family@engine"],
                "issue_files": ["new_issue/generated/audit_family.md"],
            }
        ),
        encoding="utf-8",
    )
    (root / "new_issue/generated/audit_family.md").write_text("# Generated\n", encoding="utf-8")
    (root / "new_issue/generated/bug-sprint-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "bug-sprint-v1",
                "lane_ids": ["arrow_layout"],
                "seeds": [1],
                "summary": {"run_count": 1, "fresh_candidate_bug_families": {}},
            }
        ),
        encoding="utf-8",
    )
    (root / "new_issue/generated/issue-bundles").mkdir(parents=True, exist_ok=True)
    (root / "new_issue/generated/issue-bundles/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "issue-bundle-v1",
                "summary": {
                    "family_count": 1,
                    "families": ["bundle_family@engine"],
                    "extracted_reproducer_count": 1,
                    "missing_reproducer_count": 0,
                    "compile_failure_count": 0,
                    "executed_reproducer_count": 1,
                    "nonzero_exit_count": 0,
                    "timeout_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "old_issue/README.md").write_text(
        "https://github.com/old/project/issues/1\n",
        encoding="utf-8",
    )


def _review_issue_body(*, title: str, family: str, status: str) -> str:
    return f"""# {title}

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `{family}` |
| First recorded live signal | 2026-05-28 10:00:00 CST |
| How found | DataDiffFuzz produced a repeatable latest-version mismatch |
| Status | {status} |

## Environment

```text
python: 3.12
```

## Reproducer

```python
print("ready")
```

## Actual Output

```text
wrong
```

## Expected Output

```text
right
```

## DataDiffFuzz Evidence

- Generated by project code from a latest-version run.
"""


def test_build_review_readiness_passes_minimal_project_when_target_is_met(tmp_path):
    _write_minimal_review_project(tmp_path)

    audit = build_review_readiness(
        project_root=tmp_path,
        latest_confirmation_files=[tmp_path / "experiments/latest_confirmations.json"],
        thresholds=ReviewThresholds(target_confirmed_bug_families=1),
    )

    assert audit["ready"] is True
    assert audit["summary"]["required_passed"] == audit["summary"]["required_total"]
    assert audit["summary"]["confirmed_latest_count"] == 1
    assert "DataDiffFuzz Review Readiness" in render_review_readiness_markdown(audit)


def test_build_review_readiness_reports_family_level_issue_queue(tmp_path):
    _write_minimal_review_project(tmp_path)
    new_issue = tmp_path / "new_issue"
    (new_issue / "ready.md").write_text(
        _review_issue_body(
            title="Ready",
            family="ready_family@polars",
            status="New candidate issue, not yet submitted upstream",
        ),
        encoding="utf-8",
    )
    (new_issue / "ready_support.md").write_text(
        _review_issue_body(
            title="Ready Support",
            family="ready_family@polars",
            status="New candidate issue, not yet submitted upstream",
        ),
        encoding="utf-8",
    )
    (new_issue / "dedup.md").write_text(
        _review_issue_body(
            title="Dedup",
            family="dedup_family@datafusion",
            status="New candidate issue; needs final upstream dedup check before counting as confirmed",
        ),
        encoding="utf-8",
    )
    (new_issue / "flaky.md").write_text(
        _review_issue_body(
            title="Flaky",
            family="flaky_family@duckdb",
            status="Needs stable reproducer; repeated issue-bundle attempts are flaky",
        ),
        encoding="utf-8",
    )

    audit = build_review_readiness(
        project_root=tmp_path,
        latest_confirmation_files=[tmp_path / "experiments/latest_confirmations.json"],
        thresholds=ReviewThresholds(target_confirmed_bug_families=1),
    )

    assert audit["summary"]["ready_to_submit_families"] == ["ready_family@polars"]
    assert audit["summary"]["needs_dedup_check_families"] == ["dedup_family@datafusion"]
    assert str(new_issue / "flaky.md") in audit["summary"]["needs_reproducer_or_evidence"]
    assert audit["summary"]["issue_submission_group_count"] == 2
    assert audit["summary"]["duplicate_family_draft_count"] == 1
    assert any("Submit ready unique issue families: ready_family@polars." == item for item in audit["recommendations"])
    assert any("Run final upstream duplicate checks for issue families: dedup_family@datafusion." == item for item in audit["recommendations"])
    assert any(item.startswith("Stabilize or complete reproducer/evidence") for item in audit["recommendations"])
    markdown = render_review_readiness_markdown(audit)
    assert "## Issue Submission Groups" in markdown
    assert "ready_family@polars" in markdown


def test_build_review_readiness_rejects_broken_issue_bundle_reproducers(tmp_path):
    _write_minimal_review_project(tmp_path)
    (tmp_path / "new_issue/generated/issue-bundles/manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "issue-bundle-v1",
                "summary": {
                    "family_count": 1,
                    "families": ["bundle_family@engine"],
                    "extracted_reproducer_count": 1,
                    "missing_reproducer_count": 0,
                    "compile_failure_count": 0,
                    "executed_reproducer_count": 1,
                    "nonzero_exit_count": 1,
                    "timeout_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    audit = build_review_readiness(
        project_root=tmp_path,
        latest_confirmation_files=[tmp_path / "experiments/latest_confirmations.json"],
        thresholds=ReviewThresholds(target_confirmed_bug_families=1),
    )

    criteria = {item["id"]: item for item in audit["criteria"]}
    assert audit["ready"] is False
    assert "automated_bug_detection_pipeline" in audit["summary"]["failed_required"]
    assert criteria["automated_bug_detection_pipeline"]["status"] == "fail"
    assert "nonzero exits" in criteria["automated_bug_detection_pipeline"]["evidence"]


def test_build_review_readiness_reports_confirmed_bug_gap(tmp_path):
    _write_minimal_review_project(tmp_path)

    audit = build_review_readiness(
        project_root=tmp_path,
        latest_confirmation_files=[tmp_path / "experiments/latest_confirmations.json"],
        thresholds=ReviewThresholds(target_confirmed_bug_families=2),
    )

    assert audit["ready"] is False
    assert "confirmed_bug_family_target" in audit["summary"]["failed_required"]
    assert "Need 1 more upstream-confirmed" in audit["recommendations"][0]


def test_build_review_readiness_uses_focused_sprint_wording_for_empty_fresh_queue(tmp_path):
    _write_minimal_review_project(tmp_path)

    audit = build_review_readiness(
        project_root=tmp_path,
        latest_confirmation_files=[tmp_path / "experiments/latest_confirmations.json"],
        thresholds=ReviewThresholds(target_confirmed_bug_families=2),
    )

    assert any("focused organic sprint lanes" in item for item in audit["recommendations"])


def test_build_review_readiness_requires_complete_final_experiment_protocol(tmp_path):
    _write_minimal_review_project(tmp_path)
    (tmp_path / "experiments/final_protocol.md").write_text(
        "scripts/run_final_experiments.py --track live\n",
        encoding="utf-8",
    )

    audit = build_review_readiness(
        project_root=tmp_path,
        latest_confirmation_files=[tmp_path / "experiments/latest_confirmations.json"],
        thresholds=ReviewThresholds(target_confirmed_bug_families=1),
    )

    criteria = {item["id"]: item for item in audit["criteria"]}
    assert audit["ready"] is False
    assert "final_experiment_protocol" in audit["summary"]["failed_required"]
    assert criteria["final_experiment_protocol"]["status"] == "fail"
    assert "protocol doc missing track commands" in criteria["final_experiment_protocol"]["evidence"]


def test_write_review_readiness_outputs(tmp_path):
    audit = {
        "schema_version": "review-readiness-v1",
        "generated_at": "2026-05-27T12:00:00Z",
        "ready": False,
        "summary": {
            "required_passed": 1,
            "required_total": 2,
            "confirmed_latest_count": 1,
            "target_confirmed_bug_families": 20,
        },
        "criteria": [
            {"id": "gate", "required": True, "status": "pass", "evidence": "ok"},
        ],
        "recommendations": ["do more"],
    }

    json_path, md_path = write_review_readiness_outputs(audit, output_dir=tmp_path)

    assert json_path.name == "review-readiness-20260527T120000.json"
    assert md_path.name == "review-readiness-20260527T120000.md"
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == "review-readiness-v1"
    assert "do more" in md_path.read_text(encoding="utf-8")


def test_cli_review_readiness_prints_json(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "build_review_readiness",
        lambda **kwargs: {
            "schema_version": "review-readiness-v1",
            "ready": True,
            "summary": {
                "required_passed": 1,
                "required_total": 1,
                "confirmed_latest_count": 20,
                "target_confirmed_bug_families": 20,
            },
            "criteria": [],
            "recommendations": [],
        },
    )

    args = build_parser().parse_args(["review-readiness", "--json", "--target-confirmed", "20"])

    assert args.func(args) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "review-readiness-v1"
