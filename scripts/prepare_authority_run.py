#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datadiff.util import utc_now  # noqa: E402

AUTHORITY_LAUNCHER = Path("scripts/start_closed_loop_24h_tmux.sh")
AUTHORITY_PREP_SCHEMA_VERSION = "authority-run-prep-v1"
DEFAULT_MANIFEST_INDEX = Path("reports/final-experiment-manifest-index.json")
DEFAULT_LEDGER_EVIDENCE_MANIFEST = Path("reports/experiment-final-version-ledger.json")
DEFAULT_PAPER_RUN_JOURNAL = Path("reports/paper-run-journal.jsonl")
DEFAULT_STRATEGY_SNAPSHOT = Path("reports/strategy-snapshots/final-frozen-strategy-snapshot.json")
AUTHORITY_CODE_DIRTY_PREFIXES = (
    "rust_kernel/",
    "src/",
    "scripts/",
    "tests/",
)
AUTHORITY_CODE_DIRTY_FILES = frozenset(
    {
        "AGENTS.md",
        "Cargo.lock",
        "Cargo.toml",
        "pyproject.toml",
        "pytest.ini",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "uv.lock",
    }
)


@dataclass(frozen=True, slots=True)
class AuthorityPrepResult:
    source_root: Path
    prepared_root: Path
    source_head_commit: str
    prepared_head_commit: str
    source_workspace_dirty: bool
    source_code_dirty_paths: tuple[str, ...]
    prepared_workspace_dirty: bool
    created_worktree: bool
    launcher: Path
    report_file: Path
    env_file: Path
    launch_command: str


def main() -> int:
    return prepare_with_args(parse_args())


def prepare_with_args(args: argparse.Namespace) -> int:
    result = prepare_authority_run(args)
    print(f"authority prep report: {result.report_file}")
    print(f"prepared root:         {result.prepared_root}")
    print(f"launcher env:          {result.env_file}")
    print(f"launcher script:       {result.launcher}")
    print(f"launch command:        {result.launch_command}")
    return 0


def prepare_authority_run(args: argparse.Namespace) -> AuthorityPrepResult:
    source_root = _git_toplevel(Path(str(args.project_root)).expanduser().resolve())
    launcher = Path(str(getattr(args, "launcher", AUTHORITY_LAUNCHER) or AUTHORITY_LAUNCHER))
    if launcher != AUTHORITY_LAUNCHER:
        raise SystemExit(
            f"authority preparation requires the current launcher only: {AUTHORITY_LAUNCHER}"
        )
    if not (source_root / launcher).is_file():
        raise SystemExit(f"missing authority launcher: {source_root / launcher}")

    source_head_commit = _git_stdout(source_root, "rev-parse", "HEAD")
    source_workspace_dirty = _workspace_dirty(source_root)
    source_code_dirty_paths = tuple(_source_code_dirty_paths(source_root))
    if source_code_dirty_paths and not bool(getattr(args, "allow_source_code_dirty", False)):
        preview = ", ".join(source_code_dirty_paths[:8])
        suffix = "" if len(source_code_dirty_paths) <= 8 else f", ... ({len(source_code_dirty_paths)} total)"
        raise SystemExit(
            "authority preparation would exclude uncommitted code changes from the clean "
            f"worktree: {preview}{suffix}. Commit or stash these changes, or pass "
            "--allow-source-code-dirty only for an explicit HEAD-only authority run."
        )
    prepared_root = _resolve_prepared_root(
        source_root,
        head_commit=source_head_commit,
        source_workspace_dirty=source_workspace_dirty,
        explicit_prepared_root=str(getattr(args, "prepared_root", "") or "").strip(),
        force_worktree=bool(getattr(args, "force_worktree", False)),
    )
    created_worktree = prepared_root != source_root
    if created_worktree:
        _ensure_prepared_worktree(source_root, prepared_root, head_commit=source_head_commit)
    prepared_head_commit = _git_stdout(prepared_root, "rev-parse", "HEAD")
    prepared_workspace_dirty = _workspace_dirty(prepared_root)
    if prepared_head_commit != source_head_commit:
        raise SystemExit(
            f"prepared worktree head mismatch: source={source_head_commit} prepared={prepared_head_commit}"
        )
    if prepared_workspace_dirty:
        raise SystemExit(f"prepared worktree is not clean: {prepared_root}")
    if not (prepared_root / launcher).is_file():
        raise SystemExit(f"prepared authority launcher is missing: {prepared_root / launcher}")

    stamp = utc_now().replace(":", "").replace("-", "").replace("Z", "")
    prep_dir = prepared_root / "reports" / "authority-prep"
    prep_dir.mkdir(parents=True, exist_ok=True)
    report_file = prep_dir / f"authority-prep-{stamp}.json"
    env_file = prep_dir / f"authority-prep-{stamp}.env"

    manifest_index = Path(str(getattr(args, "manifest_index", "") or DEFAULT_MANIFEST_INDEX))
    ledger_evidence_manifest = Path(
        str(getattr(args, "ledger_evidence_manifest", "") or DEFAULT_LEDGER_EVIDENCE_MANIFEST)
    )
    paper_run_journal = Path(str(getattr(args, "paper_run_journal", "") or DEFAULT_PAPER_RUN_JOURNAL))
    strategy_snapshot = Path(str(getattr(args, "strategy_snapshot", "") or DEFAULT_STRATEGY_SNAPSHOT))
    continual_learning_ledgers = str(getattr(args, "continual_learning_ledgers", "") or "").strip()
    tmux_session = str(getattr(args, "tmux_session", "") or "").strip()

    env_payload = {
        "DATADIFF_ROOT_DIR": str(prepared_root),
        "DATADIFF_FINAL_READINESS_MANIFEST_INDEX": str(manifest_index),
        "DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS": str(ledger_evidence_manifest),
        "DATADIFF_REQUIRE_CLEAN_WORKTREE": "1",
    }
    source_python = source_root / ".venv" / "bin" / "python"
    if source_python.is_file():
        env_payload["DATADIFF_PYTHON"] = str(source_python)
        env_payload["DATADIFF_PYTHONPATH"] = str(prepared_root / "src")
    if continual_learning_ledgers:
        env_payload["DATADIFF_CONTINUAL_LEARNING_LEDGERS"] = continual_learning_ledgers
    if tmux_session:
        env_payload["DATADIFF_TMUX_SESSION"] = tmux_session
    _write_env_file(env_file, env_payload)

    launch_command = (
        f"cd {shlex.quote(str(prepared_root))} && "
        f"source {shlex.quote(str(env_file))} && "
        f"bash {shlex.quote(str(launcher))}"
    )
    report_payload = {
        "schema_version": AUTHORITY_PREP_SCHEMA_VERSION,
        "prepared_at": utc_now(),
        "source_root": str(source_root),
        "prepared_root": str(prepared_root),
        "source_head_commit": source_head_commit,
        "prepared_head_commit": prepared_head_commit,
        "source_workspace_dirty": source_workspace_dirty,
        "source_code_dirty_paths": list(source_code_dirty_paths),
        "prepared_workspace_dirty": prepared_workspace_dirty,
        "created_worktree": created_worktree,
        "launcher": str(launcher),
        "launcher_path": str(prepared_root / launcher),
        "env_file": str(env_file),
        "manifest_index": str(manifest_index),
        "ledger_evidence_manifest": str(ledger_evidence_manifest),
        "paper_run_journal": str(paper_run_journal),
        "strategy_snapshot": str(strategy_snapshot),
        "continual_learning_ledgers": continual_learning_ledgers,
        "tmux_session": tmux_session,
        "launch_command": launch_command,
    }
    report_file.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AuthorityPrepResult(
        source_root=source_root,
        prepared_root=prepared_root,
        source_head_commit=source_head_commit,
        prepared_head_commit=prepared_head_commit,
        source_workspace_dirty=source_workspace_dirty,
        source_code_dirty_paths=source_code_dirty_paths,
        prepared_workspace_dirty=prepared_workspace_dirty,
        created_worktree=created_worktree,
        launcher=prepared_root / launcher,
        report_file=report_file,
        env_file=env_file,
        launch_command=launch_command,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a clean authority-run workspace and launcher env for "
            "scripts/start_closed_loop_24h_tmux.sh."
        )
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="repository root to prepare for an authority run",
    )
    parser.add_argument(
        "--prepared-root",
        default="",
        help=(
            "target clean authority worktree root; when omitted, reuse the current root if it is clean, "
            "otherwise create a sibling detached worktree"
        ),
    )
    parser.add_argument(
        "--force-worktree",
        action="store_true",
        help="always use a dedicated detached worktree even when the current root is already clean",
    )
    parser.add_argument(
        "--allow-source-code-dirty",
        action="store_true",
        help=(
            "allow authority preparation when source/scripts/tests or project config files have "
            "uncommitted changes; the prepared worktree will still run the current HEAD only"
        ),
    )
    parser.add_argument(
        "--launcher",
        default=str(AUTHORITY_LAUNCHER),
        help="launcher to prepare; only scripts/start_closed_loop_24h_tmux.sh is accepted",
    )
    parser.add_argument(
        "--manifest-index",
        default=str(DEFAULT_MANIFEST_INDEX),
        help="final experiment manifest index path to inject into the launcher env",
    )
    parser.add_argument(
        "--ledger-evidence-manifest",
        default=str(DEFAULT_LEDGER_EVIDENCE_MANIFEST),
        help="final version-ledger evidence manifest to inject into the launcher env",
    )
    parser.add_argument(
        "--paper-run-journal",
        default=str(DEFAULT_PAPER_RUN_JOURNAL),
        help="paper-run journal path recorded in the prep report",
    )
    parser.add_argument(
        "--strategy-snapshot",
        default=str(DEFAULT_STRATEGY_SNAPSHOT),
        help="strategy snapshot path recorded in the prep report",
    )
    parser.add_argument(
        "--continual-learning-ledgers",
        default="",
        help="optional comma-separated continual-learning ledgers to inject into the launcher env",
    )
    parser.add_argument(
        "--tmux-session",
        default="",
        help="optional authority tmux session override to inject into the launcher env",
    )
    return parser.parse_args()


def _resolve_prepared_root(
    source_root: Path,
    *,
    head_commit: str,
    source_workspace_dirty: bool,
    explicit_prepared_root: str,
    force_worktree: bool,
) -> Path:
    if explicit_prepared_root:
        return Path(explicit_prepared_root).expanduser().resolve()
    if not source_workspace_dirty and not force_worktree:
        return source_root
    short_head = head_commit[:12] if head_commit else "unknown"
    return source_root.parent / f"{source_root.name}-authority-{short_head}"


def _ensure_prepared_worktree(source_root: Path, prepared_root: Path, *, head_commit: str) -> None:
    if prepared_root.exists():
        if not (prepared_root / ".git").exists():
            raise SystemExit(f"prepared root exists but is not a git worktree: {prepared_root}")
        existing_head = _git_stdout(prepared_root, "rev-parse", "HEAD")
        if existing_head != head_commit:
            raise SystemExit(
                f"prepared worktree already exists at a different commit: {prepared_root} ({existing_head})"
            )
        if _workspace_dirty(prepared_root):
            raise SystemExit(f"prepared worktree already exists but is dirty: {prepared_root}")
        return
    prepared_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "add", "--detach", str(prepared_root), head_commit],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_env_file(path: Path, payload: dict[str, str]) -> None:
    lines = [
        f"export {key}={shlex.quote(str(value))}"
        for key, value in payload.items()
        if str(value).strip()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _workspace_dirty(root: Path) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(proc.stdout.strip())


def _workspace_status_paths(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=True,
        capture_output=True,
    )
    entries = [item.decode("utf-8", errors="replace") for item in proc.stdout.split(b"\0") if item]
    paths: list[str] = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        status = entry[:2]
        path = entry[3:] if len(entry) > 3 else ""
        if path:
            paths.append(path)
        if status[:1] in {"R", "C"} or status[1:2] in {"R", "C"}:
            if i + 1 < len(entries):
                paths.append(entries[i + 1])
            i += 2
        else:
            i += 1
    return paths


def _source_code_dirty_paths(root: Path) -> list[str]:
    dirty: list[str] = []
    for path in _workspace_status_paths(root):
        normalized = path.strip().lstrip("/")
        if not normalized:
            continue
        if normalized in AUTHORITY_CODE_DIRTY_FILES or normalized.startswith(AUTHORITY_CODE_DIRTY_PREFIXES):
            dirty.append(normalized)
    return sorted(dict.fromkeys(dirty))


def _git_toplevel(root: Path) -> Path:
    text = _git_stdout(root, "rev-parse", "--show-toplevel")
    if not text:
        raise SystemExit(f"not a git worktree: {root}")
    return Path(text).resolve()


def _git_stdout(root: Path, *args: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return proc.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
