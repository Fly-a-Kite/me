from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from datadiff.issue_readiness import build_issue_readiness
from datadiff.util import PROJECT_ROOT, dump_json, slugify, utc_now

ISSUE_BUNDLE_SCHEMA_VERSION = "issue-bundle-v1"
DEFAULT_ISSUE_BUNDLE_STATUSES = ("ready_to_submit", "needs_dedup_check")
DEFAULT_ISSUE_BUNDLE_DIR = PROJECT_ROOT / "new_issue" / "generated" / "issue-bundles"


def build_issue_bundle(
    *,
    latest_confirmation_files: list[Path] | None = None,
    new_issue_dir: Path | None = None,
    old_issue_dir: Path | None = None,
    generated_issue_dir: Path | None = None,
    output_dir: Path | None = None,
    statuses: list[str] | None = None,
    run_reproducers: bool = False,
    timeout_s: float = 20.0,
    repeat_count: int = 1,
    primary_per_family: bool = False,
) -> dict[str, Any]:
    output_dir = _resolve_project_path(output_dir or DEFAULT_ISSUE_BUNDLE_DIR)
    reproducer_dir = output_dir / "reproducers"
    selected_statuses = tuple(statuses or DEFAULT_ISSUE_BUNDLE_STATUSES)
    repeat_count = max(1, int(repeat_count or 1))
    readiness = build_issue_readiness(
        latest_confirmation_files=latest_confirmation_files,
        new_issue_dir=new_issue_dir,
        old_issue_dir=old_issue_dir,
        generated_issue_dir=generated_issue_dir,
        include_generated=False,
    )
    selected_issues, bundle_selection = _select_bundle_issues(
        readiness,
        selected_statuses=selected_statuses,
        primary_per_family=primary_per_family,
    )
    _prepare_reproducer_dir(reproducer_dir)
    bundled: list[dict[str, Any]] = []
    for issue in selected_issues:
        issue_path = _resolve_project_path(Path(str(issue.get("path", ""))))
        if not issue_path.is_file():
            bundled.append(_missing_issue_record(issue, issue_path))
            continue
        text = issue_path.read_text(encoding="utf-8")
        code_block = _select_python_reproducer(text)
        if code_block is None:
            bundled.append(_no_reproducer_record(issue, issue_path))
            continue
        reproducer_path = reproducer_dir / f"{slugify(issue_path.stem)}.py"
        reproducer_path.parent.mkdir(parents=True, exist_ok=True)
        source = code_block["source"].rstrip() + "\n"
        reproducer_path.write_text(source, encoding="utf-8")
        compile_result = _compile_reproducer(source, reproducer_path)
        if compile_result.get("status") != "ok":
            run_result = {"skipped": True, "reason": "compile_not_ok"}
        elif run_reproducers:
            attempts = [_run_reproducer(reproducer_path, timeout_s=timeout_s) for _ in range(repeat_count)]
            run_result = _summarize_reproducer_attempts(attempts)
        else:
            run_result = {"skipped": True}
        bundled.append(
            {
                "issue_path": _project_display_path(issue_path),
                "title": str(issue.get("title", "")),
                "family_guess": str(issue.get("family_guess", "")),
                "readiness_status": str(issue.get("readiness_status", "")),
                "priority": str(issue.get("priority", "")),
                "source_block_index": code_block["index"],
                "source_block_heading": code_block["heading"],
                "reproducer_path": _project_display_path(reproducer_path),
                "compile": compile_result,
                "run": run_result,
                "extracted": True,
                "blocking_reasons": list(issue.get("blocking_reasons", []) or []),
                "warnings": list(issue.get("warnings", []) or []),
            }
        )

    generated_at = utc_now()
    manifest_path = output_dir / "manifest.json"
    markdown_path = output_dir / "manifest.md"
    summary = _bundle_summary(bundled)
    summary.update(_bundle_selection_summary(bundle_selection))
    manifest = {
        "schema_version": ISSUE_BUNDLE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "generated_by": "datadiff issue-bundle",
        "manifest_path": _project_display_path(manifest_path),
        "markdown_path": _project_display_path(markdown_path),
        "inputs": {
            "latest_confirmation_files": readiness.get("inputs", {}).get("latest_confirmation_files", []),
            "new_issue_dir": readiness.get("inputs", {}).get("new_issue_dir", "new_issue"),
            "old_issue_dir": readiness.get("inputs", {}).get("old_issue_dir", "old_issue"),
            "generated_issue_dir": readiness.get("inputs", {}).get("generated_issue_dir", "new_issue/generated"),
            "statuses": list(selected_statuses),
            "run_reproducers": run_reproducers,
            "timeout_s": timeout_s,
            "repeat_count": repeat_count if run_reproducers else 0,
            "primary_per_family": primary_per_family,
        },
        "summary": summary,
        "issues": bundled,
        "bundle_selection": bundle_selection,
        "issue_readiness_summary": readiness.get("summary", {}),
        "submission_groups": readiness.get("submission_groups", []),
    }
    dump_json(manifest, manifest_path)
    markdown_path.write_text(render_issue_bundle_markdown(manifest), encoding="utf-8")
    return manifest


def render_issue_bundle_markdown(manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary", {})
    lines = [
        "# DataDiffFuzz Issue Bundle",
        "",
        f"- Generated at: `{manifest.get('generated_at', '')}`",
        f"- Primary per family: `{str(summary.get('bundled_primary_per_family', False)).lower()}`",
        f"- Eligible issue documents: `{summary.get('available_issue_count', summary.get('issue_count', 0))}`",
        f"- Bundled issue documents: `{summary.get('issue_count', 0)}`",
        f"- Bundled families: `{summary.get('family_count', 0)}`",
        f"- Duplicate family drafts: `{summary.get('duplicate_family_draft_count', 0)}`",
        f"- Supporting duplicate drafts skipped: `{summary.get('skipped_supporting_duplicate_count', 0)}`",
        f"- Extracted reproducers: `{summary.get('extracted_reproducer_count', 0)}`",
        f"- Compile failures: `{summary.get('compile_failure_count', 0)}`",
        f"- Executed reproducers: `{summary.get('executed_reproducer_count', 0)}`",
        f"- Executed attempts: `{summary.get('executed_reproducer_attempt_count', 0)}`",
        f"- Flaky reproducers: `{summary.get('flaky_reproducer_count', 0)}`",
        f"- Non-zero reproducer exits: `{summary.get('nonzero_exit_count', 0)}`",
        "",
        "## Family Groups",
        "",
        "| Family | Documents | Issues |",
        "| --- | ---: | --- |",
    ]
    family_groups = summary.get("family_groups", {})
    if family_groups:
        for family, paths in sorted(family_groups.items()):
            lines.append(f"| `{family}` | {len(paths)} | {', '.join(f'`{path}`' for path in paths)} |")
    else:
        lines.append("| none | 0 | - |")
    selection_groups = manifest.get("bundle_selection", {}).get("groups", [])
    if selection_groups:
        lines.extend(
            [
                "",
                "## Bundle Selection",
                "",
                "| Family | Selected Primary | Supporting Drafts | Skipped Supporting Drafts |",
                "| --- | --- | --- | --- |",
            ]
        )
        for group in selection_groups:
            supporting = ", ".join(f"`{path}`" for path in group.get("supporting_issue_paths", [])) or "-"
            skipped = (
                ", ".join(f"`{path}`" for path in group.get("skipped_supporting_issue_paths", []))
                or "-"
            )
            lines.append(
                f"| `{group.get('family', '')}` | `{group.get('selected_issue_path', '')}` | "
                f"{supporting} | {skipped} |"
            )
    lines.extend(
        [
        "",
        "## Reproducers",
        "",
        "| Issue | Family | Status | Reproducer | Compile | Run |",
        "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in manifest.get("issues", []):
        compile_status = str(item.get("compile", {}).get("status", ""))
        run = item.get("run", {})
        if run.get("skipped"):
            run_status = "skipped"
        elif run.get("timed_out"):
            run_status = "timeout"
        else:
            run_status = f"exit {run.get('returncode', '')}"
        lines.append(
            f"| `{item.get('issue_path', '')}` | `{item.get('family_guess', '')}` | "
            f"`{item.get('readiness_status', '')}` | `{item.get('reproducer_path', '')}` | "
            f"`{compile_status}` | `{run_status}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _select_bundle_issues(
    readiness: dict[str, Any],
    *,
    selected_statuses: tuple[str, ...],
    primary_per_family: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_status_set = set(selected_statuses)
    eligible_issues = [
        issue
        for issue in readiness.get("issues", [])
        if str(issue.get("readiness_status", "")) in selected_status_set
    ]
    eligible_by_path = {
        str(issue.get("path", "")): issue for issue in eligible_issues if str(issue.get("path", ""))
    }
    selected_paths: list[str] = []
    grouped_eligible_paths: set[str] = set()
    selection_groups: list[dict[str, Any]] = []

    for group in readiness.get("submission_groups", []):
        group_paths = [str(path) for path in group.get("issue_paths", []) if str(path)]
        eligible_group_paths = [path for path in group_paths if path in eligible_by_path]
        if not eligible_group_paths:
            continue
        selected_path = str(group.get("primary_issue_path", ""))
        if selected_path not in eligible_by_path:
            selected_path = eligible_group_paths[0]
        paths_to_add = [selected_path] if primary_per_family else eligible_group_paths
        _append_unique(selected_paths, paths_to_add)
        grouped_eligible_paths.update(eligible_group_paths)
        supporting_paths = [path for path in group_paths if path != selected_path]
        skipped_supporting_paths = (
            [path for path in eligible_group_paths if path != selected_path] if primary_per_family else []
        )
        selection_groups.append(
            {
                "family": str(group.get("family", "")),
                "primary_issue_path": str(group.get("primary_issue_path", "")),
                "selected_issue_path": selected_path,
                "issue_paths": group_paths,
                "eligible_issue_paths": eligible_group_paths,
                "supporting_issue_paths": supporting_paths,
                "skipped_supporting_issue_paths": skipped_supporting_paths,
                "status_counts": dict(group.get("status_counts", {}) or {}),
                "needs_dedup_check": bool(group.get("needs_dedup_check")),
            }
        )

    remaining_issues = [
        issue for issue in eligible_issues if str(issue.get("path", "")) not in grouped_eligible_paths
    ]
    if primary_per_family:
        remaining_by_family: dict[str, list[dict[str, Any]]] = {}
        for issue in remaining_issues:
            family = str(issue.get("family_guess", "")).strip() or str(issue.get("path", "")).strip()
            remaining_by_family.setdefault(family, []).append(issue)
        for family, family_issues in remaining_by_family.items():
            ordered = sorted(family_issues, key=_issue_selection_sort_key)
            selected_path = str(ordered[0].get("path", ""))
            supporting_paths = [str(issue.get("path", "")) for issue in ordered[1:] if str(issue.get("path", ""))]
            _append_unique(selected_paths, [selected_path])
            selection_groups.append(
                {
                    "family": family,
                    "primary_issue_path": selected_path,
                    "selected_issue_path": selected_path,
                    "issue_paths": [str(issue.get("path", "")) for issue in ordered],
                    "eligible_issue_paths": [str(issue.get("path", "")) for issue in ordered],
                    "supporting_issue_paths": supporting_paths,
                    "skipped_supporting_issue_paths": supporting_paths,
                    "status_counts": _status_counts(ordered),
                    "needs_dedup_check": False,
                }
            )
    else:
        _append_unique(selected_paths, [str(issue.get("path", "")) for issue in remaining_issues])

    selected_issues = [eligible_by_path[path] for path in selected_paths if path in eligible_by_path]
    skipped_supporting_issue_paths = _unique_sorted(
        path
        for group in selection_groups
        for path in group.get("skipped_supporting_issue_paths", [])
        if path
    )
    supporting_issue_paths = _unique_sorted(
        path for group in selection_groups for path in group.get("supporting_issue_paths", []) if path
    )
    return selected_issues, {
        "primary_per_family": primary_per_family,
        "selected_statuses": list(selected_statuses),
        "available_issue_count": len(eligible_issues),
        "selected_issue_count": len(selected_issues),
        "available_issue_paths": [str(issue.get("path", "")) for issue in eligible_issues],
        "selected_issue_paths": [str(issue.get("path", "")) for issue in selected_issues],
        "available_submission_group_count": len(selection_groups),
        "selected_submission_group_count": len(selection_groups),
        "primary_issue_paths": [str(group.get("selected_issue_path", "")) for group in selection_groups],
        "supporting_issue_paths": supporting_issue_paths,
        "skipped_supporting_issue_paths": skipped_supporting_issue_paths,
        "skipped_supporting_duplicate_count": len(skipped_supporting_issue_paths),
        "groups": selection_groups,
    }


def _bundle_selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "bundled_primary_per_family": bool(selection.get("primary_per_family")),
        "available_issue_count": int(selection.get("available_issue_count", 0) or 0),
        "selected_issue_count": int(selection.get("selected_issue_count", 0) or 0),
        "available_submission_group_count": int(selection.get("available_submission_group_count", 0) or 0),
        "selected_submission_group_count": int(selection.get("selected_submission_group_count", 0) or 0),
        "primary_issue_paths": list(selection.get("primary_issue_paths", []) or []),
        "supporting_issue_paths": list(selection.get("supporting_issue_paths", []) or []),
        "skipped_supporting_issue_paths": list(selection.get("skipped_supporting_issue_paths", []) or []),
        "skipped_supporting_duplicate_count": int(
            selection.get("skipped_supporting_duplicate_count", 0) or 0
        ),
    }


def _append_unique(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _issue_selection_sort_key(issue: dict[str, Any]) -> tuple[int, str]:
    return (-int(issue.get("priority_score", 0) or 0), str(issue.get("path", "")))


def _status_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for issue in issues:
        status = str(issue.get("readiness_status", ""))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _unique_sorted(values: Any) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _select_python_reproducer(text: str) -> dict[str, Any] | None:
    blocks = _markdown_code_blocks(text)
    python_blocks = [
        block
        for block in blocks
        if block["language"] in {"python", "py"} or not block["language"] and _looks_like_python(block["source"])
    ]
    if not python_blocks:
        return None
    for block in python_blocks:
        heading = block["heading"].lower()
        if "reproducer" in heading or "reproduction" in heading or "复现" in heading:
            return block
    return python_blocks[0]


def _markdown_code_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    heading = ""
    in_block = False
    language = ""
    start_heading = ""
    buffer: list[str] = []
    block_index = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
        if stripped.startswith("```"):
            if not in_block:
                in_block = True
                language = stripped[3:].strip().lower()
                start_heading = heading
                buffer = []
            else:
                blocks.append(
                    {
                        "index": block_index,
                        "language": language,
                        "heading": start_heading,
                        "source": "\n".join(buffer),
                    }
                )
                block_index += 1
                in_block = False
                language = ""
                start_heading = ""
                buffer = []
            continue
        if in_block:
            buffer.append(line)
    return blocks


def _looks_like_python(source: str) -> bool:
    markers = ("import ", "from ", "print(", "def ", "pl.", "pa.", "duckdb", "SessionContext")
    return any(marker in source for marker in markers)


def _compile_reproducer(source: str, path: Path) -> dict[str, Any]:
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        return {
            "status": "syntax_error",
            "message": str(exc),
            "line": exc.lineno,
            "offset": exc.offset,
        }
    return {"status": "ok"}


def _prepare_reproducer_dir(reproducer_dir: Path) -> None:
    if not reproducer_dir.is_dir():
        return
    for path in reproducer_dir.glob("*.py"):
        if path.is_file():
            path.unlink()


def _run_reproducer(path: Path, *, timeout_s: float) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "skipped": False,
            "timed_out": True,
            "timeout_s": timeout_s,
            "stdout": _truncate(exc.stdout or ""),
            "stderr": _truncate(exc.stderr or ""),
        }
    return {
        "skipped": False,
        "timed_out": False,
        "returncode": completed.returncode,
        "stdout": _truncate(completed.stdout),
        "stderr": _truncate(completed.stderr),
    }


def _summarize_reproducer_attempts(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    if not attempts:
        return {"skipped": True}
    first = dict(attempts[0])
    returncodes = [
        attempt.get("returncode")
        for attempt in attempts
        if not attempt.get("skipped") and not attempt.get("timed_out")
    ]
    stdout_values = [str(attempt.get("stdout", "")) for attempt in attempts if not attempt.get("skipped")]
    stderr_values = [str(attempt.get("stderr", "")) for attempt in attempts if not attempt.get("skipped")]
    timeout_count = sum(1 for attempt in attempts if attempt.get("timed_out"))
    nonzero_count = sum(
        1
        for attempt in attempts
        if not attempt.get("skipped")
        and not attempt.get("timed_out")
        and int(attempt.get("returncode", 0) or 0) != 0
    )
    consistent_returncodes = len(set(returncodes)) <= 1 and timeout_count in {0, len(attempts)}
    consistent_stdout = len(set(stdout_values)) <= 1
    consistent_stderr = len(set(stderr_values)) <= 1
    first.update(
        {
            "attempt_count": len(attempts),
            "attempts": attempts,
            "successful_attempt_count": sum(
                1
                for attempt in attempts
                if not attempt.get("skipped")
                and not attempt.get("timed_out")
                and int(attempt.get("returncode", 0) or 0) == 0
            ),
            "nonzero_attempt_count": nonzero_count,
            "timeout_attempt_count": timeout_count,
            "consistent_returncodes": consistent_returncodes,
            "consistent_stdout": consistent_stdout,
            "consistent_stderr": consistent_stderr,
            "flaky": not (consistent_returncodes and consistent_stdout and consistent_stderr),
        }
    )
    return first


def _missing_issue_record(issue: dict[str, Any], issue_path: Path) -> dict[str, Any]:
    return {
        "issue_path": _project_display_path(issue_path),
        "title": str(issue.get("title", "")),
        "family_guess": str(issue.get("family_guess", "")),
        "readiness_status": str(issue.get("readiness_status", "")),
        "priority": str(issue.get("priority", "")),
        "reproducer_path": "",
        "compile": {"status": "missing_issue_file"},
        "run": {"skipped": True},
        "extracted": False,
        "blocking_reasons": ["issue file is missing"],
        "warnings": list(issue.get("warnings", []) or []),
    }


def _no_reproducer_record(issue: dict[str, Any], issue_path: Path) -> dict[str, Any]:
    return {
        "issue_path": _project_display_path(issue_path),
        "title": str(issue.get("title", "")),
        "family_guess": str(issue.get("family_guess", "")),
        "readiness_status": str(issue.get("readiness_status", "")),
        "priority": str(issue.get("priority", "")),
        "reproducer_path": "",
        "compile": {"status": "missing_python_reproducer"},
        "run": {"skipped": True},
        "extracted": False,
        "blocking_reasons": ["no Python reproducer code block found in issue draft"],
        "warnings": list(issue.get("warnings", []) or []),
    }


def _bundle_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({str(item.get("family_guess", "")).strip() for item in items if item.get("family_guess")})
    family_groups = _family_groups(items)
    duplicate_family_groups = {
        family: paths for family, paths in family_groups.items() if len(paths) > 1
    }
    extracted = [item for item in items if item.get("extracted")]
    compile_failures = [
        item
        for item in extracted
        if str(item.get("compile", {}).get("status", "")) not in {"", "ok"}
    ]
    executed = [item for item in items if not item.get("run", {}).get("skipped")]
    nonzero = [
        item
        for item in executed
        if int(item.get("run", {}).get("nonzero_attempt_count", 0) or 0)
        or (
            not item.get("run", {}).get("timed_out")
            and int(item.get("run", {}).get("returncode", 0) or 0) != 0
        )
    ]
    timed_out = [
        item
        for item in executed
        if item.get("run", {}).get("timed_out") or int(item.get("run", {}).get("timeout_attempt_count", 0) or 0)
    ]
    flaky = [item for item in executed if item.get("run", {}).get("flaky")]
    attempt_count = sum(int(item.get("run", {}).get("attempt_count", 1) or 1) for item in executed)
    nonzero_attempt_count = sum(int(item.get("run", {}).get("nonzero_attempt_count", 0) or 0) for item in executed)
    timeout_attempt_count = sum(int(item.get("run", {}).get("timeout_attempt_count", 0) or 0) for item in executed)
    return {
        "issue_count": len(items),
        "family_count": len(families),
        "families": families,
        "family_groups": family_groups,
        "duplicate_family_draft_count": sum(len(paths) - 1 for paths in duplicate_family_groups.values()),
        "duplicate_family_draft_groups": duplicate_family_groups,
        "extracted_reproducer_count": len(extracted),
        "missing_reproducer_count": len(items) - len(extracted),
        "compile_failure_count": len(compile_failures),
        "executed_reproducer_count": len(executed),
        "executed_reproducer_attempt_count": attempt_count,
        "nonzero_exit_count": len(nonzero),
        "nonzero_exit_attempt_count": nonzero_attempt_count,
        "timeout_count": len(timed_out),
        "timeout_attempt_count": timeout_attempt_count,
        "flaky_reproducer_count": len(flaky),
    }


def _family_groups(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in items:
        family = str(item.get("family_guess", "")).strip()
        if not family:
            continue
        groups.setdefault(family, []).append(str(item.get("issue_path", "")))
    return {family: sorted(paths) for family, paths in sorted(groups.items())}


def _truncate(value: str, *, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n...<truncated>..."


def _resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _project_display_path(path: str | Path) -> str:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
