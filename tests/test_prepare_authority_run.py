from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_authority_run.py"
    spec = importlib.util.spec_from_file_location("prepare_authority_run", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _init_git_repo(root: Path) -> str:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True)
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "start_closed_loop_24h_tmux.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n",
        encoding="utf-8",
    )
    (root / ".gitignore").write_text("reports/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("clean\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt", "scripts/start_closed_loop_24h_tmux.sh", ".gitignore"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _args(module, project_root: Path, **overrides):
    data = {
        "project_root": str(project_root),
        "prepared_root": "",
        "force_worktree": False,
        "launcher": str(module.AUTHORITY_LAUNCHER),
        "manifest_index": str(module.DEFAULT_MANIFEST_INDEX),
        "ledger_evidence_manifest": str(module.DEFAULT_LEDGER_EVIDENCE_MANIFEST),
        "paper_run_journal": str(module.DEFAULT_PAPER_RUN_JOURNAL),
        "strategy_snapshot": str(module.DEFAULT_STRATEGY_SNAPSHOT),
        "continual_learning_ledgers": "",
        "tmux_session": "",
        "allow_source_code_dirty": False,
    }
    data.update(overrides)
    return type("Args", (), data)()


def test_prepare_authority_run_reuses_clean_root(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    head = _init_git_repo(project_root)

    result = module.prepare_authority_run(_args(module, project_root))

    assert result.source_root == project_root
    assert result.prepared_root == project_root
    assert result.source_head_commit == head
    assert result.prepared_head_commit == head
    assert result.created_worktree is False
    assert result.prepared_workspace_dirty is False
    report = json.loads(result.report_file.read_text(encoding="utf-8"))
    assert report["schema_version"] == "authority-run-prep-v1"
    assert report["prepared_root"] == str(project_root)
    assert report["launcher"] == str(module.AUTHORITY_LAUNCHER)
    assert "start_closed_loop_24h_tmux.sh" in report["launch_command"]
    env_text = result.env_file.read_text(encoding="utf-8")
    assert f"export DATADIFF_ROOT_DIR={str(project_root)}" in env_text
    assert "DATADIFF_FINAL_READINESS_MANIFEST_INDEX" in env_text
    assert "DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS" in env_text
    assert "DATADIFF_REQUIRE_CLEAN_WORKTREE='1'" in env_text or "DATADIFF_REQUIRE_CLEAN_WORKTREE=1" in env_text


def test_prepare_authority_run_creates_detached_clean_worktree_from_dirty_source(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    head = _init_git_repo(project_root)
    (project_root / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    result = module.prepare_authority_run(_args(module, project_root))

    assert result.source_root == project_root
    assert result.created_worktree is True
    assert result.source_workspace_dirty is True
    assert result.prepared_root != project_root
    assert result.prepared_root.exists()
    assert result.prepared_head_commit == head
    assert result.prepared_workspace_dirty is False
    proc = subprocess.run(
        ["git", "-C", str(result.prepared_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""


def test_prepare_authority_run_records_external_python_for_prepared_worktree(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)
    python_path = project_root / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")

    result = module.prepare_authority_run(_args(module, project_root))

    env_text = result.env_file.read_text(encoding="utf-8")
    assert f"DATADIFF_PYTHON={str(python_path)}" in env_text
    assert f"DATADIFF_PYTHONPATH={str(result.prepared_root / 'src')}" in env_text


def test_prepare_authority_run_rejects_dirty_source_code_by_default(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)
    src_dir = project_root / "src" / "datadiff"
    src_dir.mkdir(parents=True)
    (src_dir / "new_contract.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="would exclude uncommitted code changes"):
        module.prepare_authority_run(_args(module, project_root))


def test_prepare_authority_run_allows_explicit_head_only_dirty_code_snapshot(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    head = _init_git_repo(project_root)
    scripts_dir = project_root / "scripts"
    (scripts_dir / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = module.prepare_authority_run(
        _args(module, project_root, allow_source_code_dirty=True)
    )

    assert result.created_worktree is True
    assert result.prepared_head_commit == head
    assert result.source_code_dirty_paths == ("scripts/helper.py",)
    report = json.loads(result.report_file.read_text(encoding="utf-8"))
    assert report["source_code_dirty_paths"] == ["scripts/helper.py"]


def test_prepare_authority_run_rejects_dirty_rust_kernel_code(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)
    rust_dir = project_root / "rust_kernel" / "src"
    rust_dir.mkdir(parents=True)
    (rust_dir / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="rust_kernel/src/lib.rs"):
        module.prepare_authority_run(_args(module, project_root))


def test_prepare_authority_run_rejects_renamed_source_code_old_path(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)
    src_dir = project_root / "src" / "datadiff"
    docs_dir = project_root / "docs"
    src_dir.mkdir(parents=True)
    docs_dir.mkdir(parents=True)
    source_file = src_dir / "contract.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/datadiff/contract.py"], cwd=project_root, check=True)
    subprocess.run(["git", "commit", "-m", "add source"], cwd=project_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "mv", "src/datadiff/contract.py", "docs/contract.py"],
        cwd=project_root,
        check=True,
    )

    with pytest.raises(SystemExit, match="src/datadiff/contract.py"):
        module.prepare_authority_run(_args(module, project_root))


def test_prepare_authority_run_rejects_non_authority_launcher(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)

    with pytest.raises(SystemExit, match="requires the current launcher only"):
        module.prepare_authority_run(
            _args(module, project_root, launcher="scripts/start_closed_loop_12h_tmux.sh")
        )


def test_prepare_authority_run_records_optional_launcher_overrides(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)

    result = module.prepare_authority_run(
        _args(
            module,
            project_root,
            continual_learning_ledgers="reports/ledger-a.json,reports/ledger-b.json",
            tmux_session="authority-tmux",
        )
    )

    report = json.loads(result.report_file.read_text(encoding="utf-8"))
    assert report["continual_learning_ledgers"] == "reports/ledger-a.json,reports/ledger-b.json"
    assert report["tmux_session"] == "authority-tmux"
    env_text = result.env_file.read_text(encoding="utf-8")
    assert "DATADIFF_CONTINUAL_LEARNING_LEDGERS" in env_text
    assert "DATADIFF_TMUX_SESSION" in env_text


def test_prepare_authority_run_materializes_ignored_evidence_inputs_to_worktree(tmp_path: Path):
    module = _module()
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _init_git_repo(project_root)
    reports_dir = project_root / "reports"
    reports_dir.mkdir()
    (reports_dir / "manifest-index.json").write_text('{"manifests": []}\n', encoding="utf-8")
    (reports_dir / "ledger-manifest.json").write_text('{"ledgers": []}\n', encoding="utf-8")
    (reports_dir / "c5-ledger.json").write_text('{"entries": []}\n', encoding="utf-8")

    result = module.prepare_authority_run(
        _args(
            module,
            project_root,
            force_worktree=True,
            manifest_index="reports/manifest-index.json",
            ledger_evidence_manifest="reports/ledger-manifest.json",
            continual_learning_ledgers="reports/c5-ledger.json",
        )
    )

    assert result.created_worktree is True
    assert result.prepared_root != project_root
    assert result.materialized_input_files == (
        "reports/manifest-index.json",
        "reports/ledger-manifest.json",
        "reports/c5-ledger.json",
    )
    for relative in result.materialized_input_files:
        assert (result.prepared_root / relative).read_text(encoding="utf-8")
    report = json.loads(result.report_file.read_text(encoding="utf-8"))
    assert report["materialized_input_files"] == list(result.materialized_input_files)
    proc = subprocess.run(
        ["git", "-C", str(result.prepared_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == ""
