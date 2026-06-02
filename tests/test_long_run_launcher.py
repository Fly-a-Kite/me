import errno
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_closed_loop_12h_tmux.sh"
LEGACY_LIVE_24H_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_live_24h_tmux.sh"


def _read_status(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def _wait_until(check, *, timeout: float = 15.0, interval: float = 0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = check()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("timed out waiting for condition")


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        raise
    return True


def _tmux_session_exists(session_name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        check=False,
        capture_output=True,
    ).returncode == 0


def _launcher_output_field(output: str, prefix: str) -> str | None:
    for line in output.splitlines():
        if line.startswith(prefix):
            return line.split(prefix, 1)[1].strip()
    return None


def _long_running_command() -> str:
    return "while true; do sleep 0.2; done"


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=root, check=True, capture_output=True)
    tracked = root / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def test_legacy_live_24h_launcher_delegates_to_authority_closed_loop_launcher():
    text = LEGACY_LIVE_24H_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "start_closed_loop_24h_tmux.sh" in text
    assert "--skip-paper-journal" not in text
    assert "datadiff-live-24h" in text


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_tmux_kill_cleans_process_group_and_marks_status(tmp_path: Path):
    project_root = tmp_path / "project-root"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True)

    helper = project_root / "hold-open.sh"
    helper.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p logs
            echo $$ > logs/launcher-child.pid
            {python} - <<'PY' &
            import os
            import pathlib
            import signal
            import time

            pathlib.Path("logs/worker.pid").write_text(str(os.getpid()), encoding="utf-8")
            for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, signal.SIG_IGN)
            while True:
                time.sleep(0.2)
            PY
            wait
            """
        ).format(python=sys.executable),
        encoding="utf-8",
    )
    helper.chmod(0o755)

    session_name = f"datadiff-test-{os.getpid()}-{int(time.time() * 1000)}"
    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_TMUX_SESSION": session_name,
            "DATADIFF_COMMAND": str(helper),
            "DATADIFF_TMUX_WATCH_INTERVAL": "0.1",
            "DATADIFF_REQUIRE_CLEAN_WORKTREE": "0",
        }
    )

    try:
        started = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=project_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        status_path = None
        for line in started.stdout.splitlines():
            if line.startswith("status: "):
                status_path = Path(line.split(": ", 1)[1].strip())
                break
        assert status_path is not None

        _wait_until(lambda: _read_status(status_path).get("status") == "running")
        child_pid = _wait_until(
            lambda: int((logs_dir / "launcher-child.pid").read_text(encoding="utf-8").strip())
            if (logs_dir / "launcher-child.pid").exists()
            else None
        )
        worker_pid = _wait_until(
            lambda: int((logs_dir / "worker.pid").read_text(encoding="utf-8").strip())
            if (logs_dir / "worker.pid").exists()
            else None
        )
        assert _pid_exists(child_pid)
        assert _pid_exists(worker_pid)

        subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)

        final_status = _wait_until(
            lambda: (
                status
                if (status := _read_status(status_path)).get("status", "").startswith("interrupted_")
                else None
            ),
            timeout=20.0,
        )
        assert final_status["status"] in {"interrupted_sighup", "interrupted_sigterm"}
        assert final_status["exit_code"] in {"129", "143"}
        _wait_until(lambda: not _pid_exists(child_pid), timeout=10.0)
        _wait_until(lambda: not _pid_exists(worker_pid), timeout=10.0)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_blocks_equivalent_concurrent_launches(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    session_name = f"datadiff-test-a-{os.getpid()}-{int(time.time() * 1000)}"
    duplicate_session = f"{session_name}-duplicate"
    command = _long_running_command()
    base_env = os.environ.copy()
    base_env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_COMMAND": command,
            "DATADIFF_REQUIRE_CLEAN_WORKTREE": "0",
        }
    )

    try:
        first_env = dict(base_env)
        first_env["DATADIFF_TMUX_SESSION"] = session_name
        started = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=project_root,
            env=first_env,
            check=True,
            capture_output=True,
            text=True,
        )

        status_path = _launcher_output_field(started.stdout, "status: ")
        assert status_path is not None
        _wait_until(lambda: _read_status(Path(status_path)).get("status") == "running")
        _wait_until(lambda: _tmux_session_exists(session_name))

        second_env = dict(base_env)
        second_env["DATADIFF_TMUX_SESSION"] = duplicate_session
        blocked = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=project_root,
            env=second_env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert blocked.returncode == 3
        assert "equivalent long-run already active" in blocked.stdout
        assert session_name in blocked.stdout
        assert not _tmux_session_exists(duplicate_session)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True)
        subprocess.run(["tmux", "kill-session", "-t", duplicate_session], check=False, capture_output=True)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_allows_equivalent_concurrent_launch_with_override(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    session_name = f"datadiff-test-b-{os.getpid()}-{int(time.time() * 1000)}"
    duplicate_session = f"{session_name}-override"
    command = _long_running_command()
    base_env = os.environ.copy()
    base_env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_COMMAND": command,
            "DATADIFF_REQUIRE_CLEAN_WORKTREE": "0",
        }
    )

    try:
        first_env = dict(base_env)
        first_env["DATADIFF_TMUX_SESSION"] = session_name
        started = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=project_root,
            env=first_env,
            check=True,
            capture_output=True,
            text=True,
        )

        first_status_path = _launcher_output_field(started.stdout, "status: ")
        assert first_status_path is not None
        _wait_until(lambda: _read_status(Path(first_status_path)).get("status") == "running")
        _wait_until(lambda: _tmux_session_exists(session_name))

        second_env = dict(base_env)
        second_env["DATADIFF_TMUX_SESSION"] = duplicate_session
        second_env["DATADIFF_ALLOW_EQUIVALENT_CONCURRENT_LAUNCH"] = "1"
        override_started = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=project_root,
            env=second_env,
            check=True,
            capture_output=True,
            text=True,
        )

        second_status_path = _launcher_output_field(override_started.stdout, "status: ")
        assert second_status_path is not None
        _wait_until(lambda: _read_status(Path(second_status_path)).get("status") == "running")
        _wait_until(lambda: _tmux_session_exists(duplicate_session))
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True)
        subprocess.run(["tmux", "kill-session", "-t", duplicate_session], check=False, capture_output=True)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_blocks_dirty_authority_workspace(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    _init_git_repo(project_root)
    (project_root / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_TMUX_SESSION": f"datadiff-test-dirty-{os.getpid()}",
            "DATADIFF_COMMAND": _long_running_command(),
        }
    )

    blocked = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=project_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert blocked.returncode == 4
    assert "requires a clean git worktree" in blocked.stdout


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_writes_freeze_snapshot_for_clean_authority_run(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    _init_git_repo(project_root)
    session_name = f"datadiff-test-freeze-{os.getpid()}-{int(time.time() * 1000)}"
    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_TMUX_SESSION": session_name,
            "DATADIFF_COMMAND": _long_running_command(),
        }
    )

    try:
        started = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            cwd=project_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        status_path = Path(_launcher_output_field(started.stdout, "status: "))
        _wait_until(lambda: _read_status(status_path).get("status") == "running")
        status = _read_status(status_path)
        freeze_manifest = Path(status["freeze_manifest"])
        assert freeze_manifest.exists()
        payload = freeze_manifest.read_text(encoding="utf-8")
        assert '"workspace_dirty": false' in payload
        assert ".pip-freeze.txt" in payload
        assert ".launcher-env.txt" in payload
        assert ".strategy-snapshot.json" in payload
        assert ".strategy-learning.json" in payload
        assert status["strategy_snapshot"].endswith(".strategy-snapshot.json")
        assert status["strategy_learning"].endswith(".strategy-learning.json")
        assert Path(status["strategy_snapshot"]).is_file()
        assert Path(status["strategy_learning"]).is_file()
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_records_post_run_evidence_artifacts(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    _init_git_repo(project_root)
    session_name = f"datadiff-test-postrun-{os.getpid()}-{int(time.time() * 1000)}"

    manifest_path = project_root / "runs" / "experiment-finished.json"
    command_script = project_root / "emit-manifest.sh"
    command_script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p "{project_root / 'runs'}"
            cat > "{manifest_path}" <<'JSON'
            {{"runs":[]}}
            JSON
            echo "experiment manifest: {manifest_path}"
            """
        ),
        encoding="utf-8",
    )
    command_script.chmod(0o755)

    hook_script = project_root / "post-run-hook.sh"
    hook_script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail
            mkdir -p "{project_root / 'reports' / 'classify-run-experiment-finished'}"
            summary_md="{project_root / 'reports' / 'experiment-summary-experiment-finished.md'}"
            summary_csv="{project_root / 'reports' / 'experiment-summary-experiment-finished.csv'}"
            aggregate_csv="{project_root / 'reports' / 'experiment-summary-experiment-finished-aggregates.csv'}"
            analysis_md="{project_root / 'reports' / 'experiment-analysis-experiment-finished.md'}"
            analysis_csv="{project_root / 'reports' / 'experiment-analysis-experiment-finished.csv'}"
            methodology_md="{project_root / 'reports' / 'methodology-report-experiment-finished.md'}"
            methodology_json="{project_root / 'reports' / 'methodology-report-experiment-finished.json'}"
            classify_dir="{project_root / 'reports' / 'classify-run-experiment-finished'}"
            printf 'summary\\n' > "${{summary_md}}"
            printf 'summary\\n' > "${{summary_csv}}"
            printf 'aggregate\\n' > "${{aggregate_csv}}"
            printf 'analysis\\n' > "${{analysis_md}}"
            printf 'analysis\\n' > "${{analysis_csv}}"
            printf 'methodology\\n' > "${{methodology_md}}"
            printf '{{"ok": true}}\\n' > "${{methodology_json}}"
            printf '{{"offline_buckets": {{}}}}\\n' > "${{classify_dir}}/run-a.json"
            echo "post_run_status=ok"
            echo "experiment_manifest=${{DATADIFF_POST_RUN_MANIFEST}}"
            echo "experiment_summary_markdown=${{summary_md}}"
            echo "experiment_summary_csv=${{summary_csv}}"
            echo "experiment_summary_aggregate_csv=${{aggregate_csv}}"
            echo "experiment_analysis_markdown=${{analysis_md}}"
            echo "experiment_analysis_csv=${{analysis_csv}}"
            echo "methodology_report_markdown=${{methodology_md}}"
            echo "methodology_report_json=${{methodology_json}}"
            echo "classify_run_dir=${{classify_dir}}"
            echo "classify_run_count=1"
            """
        ),
        encoding="utf-8",
    )
    hook_script.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_TMUX_SESSION": session_name,
            "DATADIFF_COMMAND": str(command_script),
            "DATADIFF_POST_RUN_EVIDENCE_HOOK": str(hook_script),
            "DATADIFF_REQUIRE_CLEAN_WORKTREE": "0",
        }
    )

    started = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    status_path = Path(_launcher_output_field(started.stdout, "status: "))
    final_status = _wait_until(
        lambda: (
            status
            if (status := _read_status(status_path)).get("status") == "completed"
            else None
        ),
        timeout=20.0,
    )

    assert final_status["post_run_status"] == "ok"
    assert final_status["experiment_manifest"] == str(manifest_path)
    assert Path(final_status["experiment_summary_markdown"]).is_file()
    assert Path(final_status["experiment_summary_csv"]).is_file()
    assert Path(final_status["experiment_summary_aggregate_csv"]).is_file()
    assert Path(final_status["experiment_analysis_markdown"]).is_file()
    assert Path(final_status["experiment_analysis_csv"]).is_file()
    assert Path(final_status["methodology_report_markdown"]).is_file()
    assert Path(final_status["methodology_report_json"]).is_file()
    assert Path(final_status["classify_run_dir"]).is_dir()
    assert final_status["classify_run_count"] == "1"
    assert (Path(final_status["classify_run_dir"]) / "run-a.json").is_file()
