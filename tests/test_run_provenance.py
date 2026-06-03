from __future__ import annotations

import subprocess
from pathlib import Path

from datadiff import run_provenance


def _init_git_repo(root: Path, *, message: str) -> str:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True)
    (root / "tracked.txt").write_text(message + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def test_run_provenance_prefers_datadiff_root_dir_for_git_state(tmp_path, monkeypatch):
    default_root = tmp_path / "default-root"
    launcher_root = tmp_path / "launcher-root"
    default_root.mkdir()
    launcher_root.mkdir()
    default_commit = _init_git_repo(default_root, message="default")
    launcher_commit = _init_git_repo(launcher_root, message="launcher")

    monkeypatch.setattr(run_provenance, "PROJECT_ROOT", default_root)
    monkeypatch.setenv("DATADIFF_ROOT_DIR", str(launcher_root))

    assert run_provenance.current_workspace_git_commit() == launcher_commit
    assert run_provenance.current_workspace_git_commit() != default_commit

