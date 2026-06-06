import json

import datadiff.issue_bundle as issue_bundle
from datadiff.issue_bundle import build_issue_bundle, render_issue_bundle_markdown


def _write_dirs(tmp_path):
    new_issue = tmp_path / "new_issue"
    generated = new_issue / "generated"
    old_issue = tmp_path / "old_issue"
    latest = tmp_path / "latest_confirmations.json"
    generated.mkdir(parents=True)
    old_issue.mkdir()
    latest.write_text(json.dumps({"confirmations": []}), encoding="utf-8")
    (generated / "manifest.json").write_text(json.dumps({"candidate_bug_families": []}), encoding="utf-8")
    return latest, new_issue, old_issue, generated


def _issue_doc(*, title: str, family: str, status: str, code: str) -> str:
    return f"""# {title}

## Discovery Record

| Item | Value |
| --- | --- |
| Project family labels observed | `{family}` |
| First recorded live signal | 2026-05-26 10:00:00 CST |
| How found | DataDiffFuzz generated a latest-version differential mismatch |
| Status | {status} |

## Environment

```text
python: 3.12
```

## Reproducer

```python
{code}
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

- Generated evidence exists in the project.
"""


def test_build_issue_bundle_extracts_and_runs_reproducers(tmp_path):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    (new_issue / "ready.md").write_text(
        _issue_doc(
            title="Ready",
            family="ready_family@pyarrow",
            status="New candidate issue, not yet submitted upstream",
            code='print("bundle-ok")',
        ),
        encoding="utf-8",
    )
    (new_issue / "ready_support.md").write_text(
        _issue_doc(
            title="Ready Support",
            family="ready_family@pyarrow",
            status="New candidate issue, not yet submitted upstream",
            code='print("bundle-support")',
        ),
        encoding="utf-8",
    )
    (new_issue / "dedup.md").write_text(
        _issue_doc(
            title="Dedup",
            family="dedup_family@datafusion",
            status="New candidate issue; needs final upstream dedup check before counting as confirmed",
            code='print("dedup-ok")',
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle"

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=output_dir,
        run_reproducers=True,
        timeout_s=5,
    )

    assert manifest["schema_version"] == "issue-bundle-v1"
    assert manifest["summary"]["issue_count"] == 3
    assert manifest["summary"]["family_count"] == 2
    assert manifest["summary"]["duplicate_family_draft_count"] == 1
    assert sorted(path.split("/")[-1] for path in manifest["summary"]["duplicate_family_draft_groups"]["ready_family@pyarrow"]) == [
        "ready.md",
        "ready_support.md",
    ]
    assert manifest["summary"]["extracted_reproducer_count"] == 3
    assert manifest["summary"]["compile_failure_count"] == 0
    assert manifest["summary"]["executed_reproducer_count"] == 3
    assert manifest["issue_readiness_summary"]["submission_group_count"] == 2
    assert any(group["family"] == "ready_family@pyarrow" and group["issue_count"] == 2 for group in manifest["submission_groups"])
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "manifest.md").is_file()
    assert (output_dir / "reproducers" / "ready.py").read_text(encoding="utf-8") == 'print("bundle-ok")\n'
    assert any("bundle-ok" in item["run"]["stdout"] for item in manifest["issues"])
    markdown = render_issue_bundle_markdown(manifest)
    assert "DataDiffFuzz Issue Bundle" in markdown
    assert "Duplicate family drafts" in markdown
    assert "ready_family@pyarrow" in markdown


def test_issue_bundle_primary_per_family_skips_supporting_duplicate_reproducers(tmp_path):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    (new_issue / "ready.md").write_text(
        _issue_doc(
            title="Ready",
            family="ready_family@pyarrow",
            status="New candidate issue, not yet submitted upstream",
            code='print("bundle-ok")',
        ),
        encoding="utf-8",
    )
    (new_issue / "ready_support.md").write_text(
        _issue_doc(
            title="Ready Support",
            family="ready_family@pyarrow",
            status="New candidate issue, not yet submitted upstream",
            code='print("bundle-support")',
        ),
        encoding="utf-8",
    )
    (new_issue / "dedup.md").write_text(
        _issue_doc(
            title="Dedup",
            family="dedup_family@datafusion",
            status="New candidate issue; needs final upstream dedup check before counting as confirmed",
            code='print("dedup-ok")',
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bundle"
    stale_reproducer = output_dir / "reproducers" / "ready_support.py"
    stale_reproducer.parent.mkdir(parents=True)
    stale_reproducer.write_text('print("stale")\n', encoding="utf-8")

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=output_dir,
        run_reproducers=True,
        primary_per_family=True,
    )

    summary = manifest["summary"]
    assert summary["bundled_primary_per_family"] is True
    assert summary["available_issue_count"] == 3
    assert summary["selected_issue_count"] == 2
    assert summary["issue_count"] == 2
    assert summary["family_count"] == 2
    assert summary["skipped_supporting_duplicate_count"] == 1
    assert summary["skipped_supporting_issue_paths"] == [str(new_issue / "ready_support.md")]
    assert manifest["issue_readiness_summary"]["duplicate_family_draft_count"] == 1
    ready_group = next(group for group in manifest["submission_groups"] if group["family"] == "ready_family@pyarrow")
    assert ready_group["supporting_issue_paths"] == [str(new_issue / "ready_support.md")]
    selection_group = next(
        group for group in manifest["bundle_selection"]["groups"] if group["family"] == "ready_family@pyarrow"
    )
    assert selection_group["selected_issue_path"] == str(new_issue / "ready.md")
    assert selection_group["skipped_supporting_issue_paths"] == [str(new_issue / "ready_support.md")]
    assert (output_dir / "reproducers" / "ready.py").is_file()
    assert (output_dir / "reproducers" / "dedup.py").is_file()
    assert not stale_reproducer.exists()
    assert all("bundle-support" not in item["run"].get("stdout", "") for item in manifest["issues"])
    markdown = render_issue_bundle_markdown(manifest)
    assert "Primary per family: `true`" in markdown
    assert "Skipped Supporting Drafts" in markdown


def test_issue_bundle_records_missing_and_syntax_error_reproducers(tmp_path):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    (new_issue / "missing.md").write_text(
        _issue_doc(
            title="Missing",
            family="missing_family@polars",
            status="New candidate issue, not yet submitted upstream",
            code="",
        ).replace("```python\n\n```", "```text\nno python here\n```"),
        encoding="utf-8",
    )
    (new_issue / "bad.md").write_text(
        _issue_doc(
            title="Bad",
            family="bad_family@polars",
            status="New candidate issue, not yet submitted upstream",
            code="def broken(:\n    pass",
        ),
        encoding="utf-8",
    )

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=tmp_path / "bundle",
    )

    by_path = {item["issue_path"].split("/")[-1]: item for item in manifest["issues"]}
    assert by_path["missing.md"]["compile"]["status"] == "missing_python_reproducer"
    assert by_path["bad.md"]["compile"]["status"] == "syntax_error"
    assert manifest["summary"]["missing_reproducer_count"] == 1
    assert manifest["summary"]["compile_failure_count"] == 1


def test_issue_bundle_uses_referenced_python_reproducer_and_accepts_expected_assertion_failure(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(issue_bundle, "PROJECT_ROOT", tmp_path)
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    repro = tmp_path / "bugs" / "bug_demo" / "standalone_reproducer.py"
    repro.parent.mkdir(parents=True)
    repro.write_text(
        'print("observed wrong result")\nassert False, "bug reproduced"\n',
        encoding="utf-8",
    )
    (new_issue / "submitted.md").write_text(
        _issue_doc(
            title="Submitted",
            family="submitted_family@engine",
            status="Submitted upstream as bug",
            code="",
        )
        .replace("```python\n\n```", "")
        + "\n- Upstream issue: https://github.com/example/project/issues/1\n"
        + "\n- Standalone reproducer: `bugs/bug_demo/standalone_reproducer.py`\n",
        encoding="utf-8",
    )

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=tmp_path / "bundle",
        statuses=["already_submitted_or_confirmed"],
        run_reproducers=True,
    )

    item = manifest["issues"][0]
    assert item["source_path"] == "bugs/bug_demo/standalone_reproducer.py"
    assert item["compile"]["status"] == "ok"
    assert item["run"]["returncode"] != 0
    assert item["run"]["expected_failure_reproduced"] is True
    assert manifest["summary"]["extracted_reproducer_count"] == 1
    assert manifest["summary"]["expected_failure_reproducer_count"] == 1
    assert manifest["summary"]["nonzero_exit_count"] == 0
    assert manifest["summary"]["nonzero_exit_attempt_count"] == 0


def test_issue_bundle_treats_fixed_upstream_not_reproduced_as_nonblocking(tmp_path):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    latest.write_text(
        json.dumps(
            {
                "confirmations": [
                    {
                        "family": "fixed_family@engine",
                        "issue_url": "https://github.com/example/project/issues/2",
                        "upstream_status": "fixed_upstream",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (new_issue / "fixed.md").write_text(
        _issue_doc(
            title="Fixed",
            family="fixed_family@engine",
            status="Submitted upstream as https://github.com/example/project/issues/2",
            code='print("No mismatch reproduced.")\nraise SystemExit(1)',
        ),
        encoding="utf-8",
    )

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=tmp_path / "bundle",
        statuses=["already_submitted_or_confirmed"],
        run_reproducers=True,
    )

    assert manifest["summary"]["fixed_upstream_not_reproduced_count"] == 1
    assert manifest["summary"]["fixed_upstream_not_reproduced_attempt_count"] == 1
    assert manifest["summary"]["nonzero_exit_count"] == 0
    assert manifest["summary"]["nonzero_exit_attempt_count"] == 0


def test_issue_bundle_does_not_run_syntax_error_reproducer(tmp_path):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    (new_issue / "bad.md").write_text(
        _issue_doc(
            title="Bad",
            family="bad_family@polars",
            status="New candidate issue, not yet submitted upstream",
            code="def broken(:\n    pass",
        ),
        encoding="utf-8",
    )

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=tmp_path / "bundle",
        run_reproducers=True,
    )

    assert manifest["summary"]["compile_failure_count"] == 1
    assert manifest["summary"]["executed_reproducer_count"] == 0
    assert manifest["issues"][0]["run"] == {"skipped": True, "reason": "compile_not_ok"}


def test_issue_bundle_repeat_runs_capture_flaky_output(tmp_path, monkeypatch):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    (new_issue / "ready.md").write_text(
        _issue_doc(
            title="Ready",
            family="ready_family@pyarrow",
            status="New candidate issue, not yet submitted upstream",
            code='print("ready")',
        ),
        encoding="utf-8",
    )
    attempts = [
        {"skipped": False, "timed_out": False, "returncode": 0, "stdout": "first\n", "stderr": ""},
        {"skipped": False, "timed_out": False, "returncode": 0, "stdout": "second\n", "stderr": ""},
    ]

    monkeypatch.setattr(
        "datadiff.issue_bundle._run_reproducer",
        lambda path, *, timeout_s: attempts.pop(0),
    )

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=tmp_path / "bundle",
        run_reproducers=True,
        repeat_count=2,
    )

    run = manifest["issues"][0]["run"]
    assert run["attempt_count"] == 2
    assert run["successful_attempt_count"] == 2
    assert run["consistent_returncodes"] is True
    assert run["consistent_stdout"] is False
    assert run["flaky"] is True
    assert manifest["summary"]["executed_reproducer_count"] == 1
    assert manifest["summary"]["executed_reproducer_attempt_count"] == 2
    assert manifest["summary"]["flaky_reproducer_count"] == 1


def test_issue_bundle_status_filter_excludes_non_selected(tmp_path):
    latest, new_issue, old_issue, generated = _write_dirs(tmp_path)
    (new_issue / "ready.md").write_text(
        _issue_doc(
            title="Ready",
            family="ready_family@pyarrow",
            status="New candidate issue, not yet submitted upstream",
            code='print("ready")',
        ),
        encoding="utf-8",
    )
    (new_issue / "dedup.md").write_text(
        _issue_doc(
            title="Dedup",
            family="dedup_family@datafusion",
            status="New candidate issue; needs final upstream dedup check before counting as confirmed",
            code='print("dedup")',
        ),
        encoding="utf-8",
    )

    manifest = build_issue_bundle(
        latest_confirmation_files=[latest],
        new_issue_dir=new_issue,
        old_issue_dir=old_issue,
        generated_issue_dir=generated,
        output_dir=tmp_path / "bundle",
        statuses=["ready_to_submit"],
    )

    assert manifest["summary"]["issue_count"] == 1
    assert manifest["issues"][0]["issue_path"].endswith("ready.md")
