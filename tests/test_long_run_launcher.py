import errno
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from importlib.metadata import version
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_closed_loop_12h_tmux.sh"
AUTHORITY_24H_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_closed_loop_24h_tmux.sh"
DISCOVERY_LONGHAUL_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "start_discovery_longhaul_tmux.sh"
)
FINAL_LIVE_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "start_final_live_campaigns_tmux.sh"
)


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


def test_only_authority_closed_loop_24h_launcher_is_exposed():
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    text = AUTHORITY_24H_SCRIPT_PATH.read_text(encoding="utf-8")

    assert not (scripts_dir / "start_live_24h_tmux.sh").exists()
    assert "start_closed_loop_12h_tmux.sh" in text
    assert "datadiff-closed-loop-24h-authority" in text
    assert "datadiff-live-24h" not in text
    assert "--skip-paper-journal" not in text


def test_authority_closed_loop_launcher_uses_final_adaptive_learning_contract():
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    for token in [
        "DATADIFF_ADAPTIVE_LEARNING_WEIGHT",
        "DATADIFF_SCHEDULER_ANNEALING_TEMPERATURE",
        "DATADIFF_SCHEDULER_ANNEALING_DECAY",
        "DATADIFF_SCHEDULER_ANNEALING_MIN_TEMPERATURE",
        "DATADIFF_CONTINUAL_LEARNING_LEDGERS",
        "--adaptive-learning-weight",
        "--scheduler-annealing-temperature",
        "--scheduler-annealing-decay",
        "--scheduler-annealing-min-temperature",
        "--persist-closed-loop-state",
        "FREEZE_ADAPTIVE_LEARNING_WEIGHT",
        "FREEZE_SCHEDULER_ANNEALING_TEMPERATURE",
        "FREEZE_CONTINUAL_LEARNING_LEDGERS",
        '"adaptive_config"',
        '"persist_closed_loop_state"',
        "adaptive_learning_weight=%s",
        "scheduler_annealing_temperature=%s",
        "continual_learning_ledgers=%s",
        "persist_closed_loop_state=1",
        "datadiff.cli final-readiness",
        "FINAL_READINESS_COMMAND+=(--manifest",
        "DATADIFF_FINAL_READINESS_MANIFEST_INDEX",
        "DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS",
        "DATADIFF_FINAL_READINESS_FAIL_ON_MISSING",
        "DATADIFF_PYTHON",
        "DATADIFF_PYTHONPATH",
        "_python_cmd",
        "_final_readiness_command",
        "--manifest-index",
        "--extra-manifest",
        "--fail-on-missing",
        "final_readiness_markdown",
        "final_readiness_json",
        "final_readiness_manifest_index",
        "final_readiness_extra_manifests",
    ]:
        assert token in text

    assert "_build_experiment_command()" in text
    assert 'if [[ -n "${CONTINUAL_LEARNING_LEDGERS}" ]]; then' in text
    assert 'EXPERIMENT_COMMAND+=(--continual-learning-ledgers "${CONTINUAL_LEARNING_LEDGERS}")' in text


def test_authority_24h_launcher_inherits_final_adaptive_defaults():
    text = AUTHORITY_24H_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'DATADIFF_ADAPTIVE_LEARNING_WEIGHT:-0.75' in text
    assert 'DATADIFF_SCHEDULER_ANNEALING_TEMPERATURE:-0.35' in text
    assert 'DATADIFF_SCHEDULER_ANNEALING_DECAY:-0.985' in text
    assert 'DATADIFF_SCHEDULER_ANNEALING_MIN_TEMPERATURE:-0.02' in text
    assert 'DATADIFF_CONTINUAL_LEARNING_LEDGERS:-' in text
    assert (
        'DATADIFF_FINAL_READINESS_MANIFEST_INDEX:-reports/final-experiment-manifest-index.json'
        in text
    )
    assert 'DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS:-' in text
    assert 'DATADIFF_FINAL_READINESS_FAIL_ON_MISSING:-0' in text
    assert 'start_closed_loop_12h_tmux.sh' in text


def test_discovery_longhaul_launcher_targets_high_yield_campaign_batches():
    text = DISCOVERY_LONGHAUL_SCRIPT_PATH.read_text(encoding="utf-8")

    for token in [
        "DATADIFF_DISCOVERY_LONGHAUL_DURATION:-12h",
        "DATADIFF_DISCOVERY_LONGHAUL_MAX_CONCURRENT",
        "DATADIFF_DISCOVERY_LONGHAUL_LOAD_LIMIT",
        "active_discovery_campaigns",
        "mktemp",
        "^([^[:space:]]*\\/)?python[0-9.]*",
        "discovery-campaign",
        "--candidate-recheck-count",
        "--candidate-pipeline-recheck-attempts",
        "discovery-campaign-aggregate",
        "DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_LATEST_TARGETS",
        "DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_SOURCE",
        "DATADIFF_DISCOVERY_LONGHAUL_FREEZE_MANIFEST",
        "DATADIFF_RUN_PROVENANCE_FREEZE_MANIFEST",
        "final-bug-discovery-freeze-v1",
        "target-version-audit",
        "target_version_audit",
        "pandas_targeted_boundaries",
        "polars_targeted_boundaries",
        "datafusion_targeted_boundaries",
        "chdb_targeted_boundaries",
        "orthogonal_stress",
        "embedded_sql,duckdb_storage",
        "common_api_workflow,cross_family",
        "--skip-run-report",
    ]:
        assert token in text

    printed = subprocess.run(
        ["bash", str(DISCOVERY_LONGHAUL_SCRIPT_PATH), "--print-config"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "duration=12h" in printed.stdout
    assert "candidate_pipeline_recheck_attempts=3" in printed.stdout
    assert "require_latest_targets=1" in printed.stdout
    assert "seed_start=31000001" in printed.stdout
    assert "target_version_audit=reports/target-version-audits/discovery-longhaul-" in printed.stdout
    assert "lane_groups=pandas_targeted_boundaries;" in printed.stdout
    assert "chdb_targeted_boundaries;orthogonal_stress;" in printed.stdout


def test_discovery_longhaul_dry_run_freezes_protocol_and_source_provenance(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    _init_git_repo(project_root)
    canonical = project_root / "experiments" / "canonical_confirmed_bug_corpus" / "v2"
    canonical.mkdir(parents=True)
    (canonical / "manifest.json").write_text(
        json.dumps({"confirmed_roots": [{"root_id": f"root-{index}"} for index in range(9)]}) + "\n",
        encoding="utf-8",
    )
    cached_audit = project_root / "cached-target-version-audit.json"
    cached_audit.write_text(
        json.dumps(
            {
                "summary": {"all_target_packages_up_to_date": True},
                "target_packages": [
                    {
                        "package": "pip",
                        "installed_version": version("pip"),
                        "latest_version": version("pip"),
                        "latest_source": "test",
                        "up_to_date": True,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_id = f"freeze-test-{os.getpid()}"
    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_PYTHON": sys.executable,
            "DATADIFF_DISCOVERY_LONGHAUL_RUN_ID": run_id,
            "DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_LATEST_TARGETS": "1",
            "DATADIFF_DISCOVERY_LONGHAUL_TARGET_VERSION_AUDIT_SOURCE": str(cached_audit),
            "DATADIFF_DISCOVERY_LONGHAUL_REQUIRE_CLEAN_WORKTREE": "0",
        }
    )

    completed = subprocess.run(
        ["bash", str(DISCOVERY_LONGHAUL_SCRIPT_PATH), "--dry-run"],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    freeze_manifest = (
        project_root
        / "reports"
        / "discovery-longhaul-provenance"
        / f"discovery-longhaul-{run_id}-freeze-manifest.json"
    )
    payload = json.loads(freeze_manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "final-bug-discovery-freeze-v1"
    assert payload["objective"]["confirmed_root_count_at_freeze"] == 9
    assert payload["objective"]["remaining_root_gap_at_freeze"] == 21
    assert payload["design"]["no_favorable_early_stopping"] is True
    assert payload["design"]["seed_start"] == 31000001
    assert "chdb_targeted_boundaries" in payload["design"]["lane_specs"]
    assert "orthogonal_stress" in payload["design"]["lane_specs"]
    assert payload["provenance"]["source_tree_sha256"]
    assert payload["provenance"]["target_version_audit_sha256"]
    assert payload["protocol_sha256"]
    assert "discovery freeze manifest:" in completed.stdout


def test_final_live_launcher_shards_campaigns_with_authority_provenance():
    text = FINAL_LIVE_SCRIPT_PATH.read_text(encoding="utf-8")

    for token in [
        "--live-campaign",
        "--skip-paper-journal",
        "--reset-manifest-index",
        "DATADIFF_RUN_PROVENANCE_AUTHORITY=1",
        "DATADIFF_RUN_PROVENANCE_FREEZE_INTENT=1",
        "DATADIFF_RUN_PROVENANCE_LATEST_CODE_CLAIM=1",
        "DATADIFF_FINAL_LIVE_REQUIRE_LATEST_TARGETS",
        "DATADIFF_RUN_PROVENANCE_TARGET_VERSION_AUDIT",
        "latest_live_authority_24h",
        "target-version-audit",
        "reports/final-live-indexes",
        "reports/final-live-provenance",
    ]:
        assert token in text

    printed = subprocess.run(
        ["bash", str(FINAL_LIVE_SCRIPT_PATH), "--print-config"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "duration=24h" in printed.stdout
    assert "require_latest_targets=1" in printed.stdout
    assert "target_version_audit=reports/final-live-provenance/final-live-" in printed.stdout
    assert "campaigns=" in printed.stdout
    assert "embedded_sql_cross:live_duckdb_issue_focus" in printed.stdout


def test_final_live_launcher_dry_run_writes_importable_campaign_artifacts(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    _init_git_repo(project_root)
    snapshot = project_root / "strategy-snapshot.json"
    snapshot.write_text("{}\n", encoding="utf-8")
    run_id = f"testrun-{os.getpid()}"
    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_PYTHON": sys.executable,
            "DATADIFF_FINAL_LIVE_CAMPAIGNS": "arrow_cross:live_arrow",
            "DATADIFF_FINAL_LIVE_RUN_ID": run_id,
            "DATADIFF_FINAL_LIVE_STRATEGY_SNAPSHOT": str(snapshot),
            "DATADIFF_FINAL_LIVE_REQUIRE_CLEAN_WORKTREE": "0",
            "DATADIFF_FINAL_LIVE_REQUIRE_LATEST_TARGETS": "0",
        }
    )

    dry_run = subprocess.run(
        ["bash", str(FINAL_LIVE_SCRIPT_PATH), "--dry-run"],
        cwd=project_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "dry-run: tmux new-session" in dry_run.stdout
    provenance_dir = project_root / "reports" / "final-live-provenance"
    freeze_manifest = provenance_dir / f"final-live-{run_id}-freeze-manifest.json"
    campaign_table = provenance_dir / f"final-live-{run_id}-campaigns.tsv"
    campaign_script = provenance_dir / f"final-live-{run_id}-arrow-cross-live-arrow.sh"
    payload = json.loads(freeze_manifest.read_text(encoding="utf-8"))

    assert payload["campaigns"] == ["arrow_cross:live_arrow"]
    assert payload["run_id"] == run_id
    assert payload["duration"] == "24h"
    assert payload["target_version_audit"].endswith(f"final-live-{run_id}-target-version-audit.json")
    assert payload["git_commit"]
    assert "arrow_cross:live_arrow" in campaign_table.read_text(encoding="utf-8")
    campaign_script_text = campaign_script.read_text(encoding="utf-8")
    assert "--live-campaign arrow_cross:live_arrow" in campaign_script_text
    assert "--skip-paper-journal" in campaign_script_text
    assert "--reset-manifest-index" in campaign_script_text


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
def test_closed_loop_launcher_uses_external_python_without_root_venv(tmp_path: Path):
    project_root = tmp_path / "project-root"
    fake_package = project_root / "src" / "datadiff"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "cli.py").write_text(
        textwrap.dedent(
            """\
            import json
            import os
            import sys
            from pathlib import Path

            Path("logs/fake-cli.json").write_text(
                json.dumps(
                    {
                        "argv": sys.argv,
                        "executable": sys.executable,
                        "pythonpath": os.environ.get("PYTHONPATH", ""),
                    },
                    sort_keys=True,
                )
                + "\\n",
                encoding="utf-8",
            )
            """
        ),
        encoding="utf-8",
    )
    session_name = f"datadiff-test-python-{os.getpid()}-{int(time.time() * 1000)}"
    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_TMUX_SESSION": session_name,
            "DATADIFF_PYTHON": sys.executable,
            "DATADIFF_PYTHONPATH": str(project_root / "src"),
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
        status_path = Path(_launcher_output_field(started.stdout, "status: "))
        final_status = _wait_until(
            lambda: (
                status
                if (status := _read_status(status_path)).get("status") == "completed"
                else None
            ),
            timeout=20.0,
        )
        payload = json.loads((project_root / "logs" / "fake-cli.json").read_text(encoding="utf-8"))

        assert final_status["post_run_status"] == "skipped"
        assert payload["executable"] == sys.executable
        assert payload["pythonpath"].split(os.pathsep)[0] == str(project_root / "src")
        assert payload["argv"][0].replace(os.sep, "/").endswith("src/datadiff/cli.py")
        assert payload["argv"][1] == "experiment"
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True)


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
        assert "final_readiness_fail_on_missing" in status
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=False, capture_output=True)


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_closed_loop_launcher_records_final_readiness_index_config_in_freeze_snapshot(tmp_path: Path):
    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True)
    _init_git_repo(project_root)
    session_name = f"datadiff-test-readiness-freeze-{os.getpid()}-{int(time.time() * 1000)}"
    env = os.environ.copy()
    env.update(
        {
            "DATADIFF_ROOT_DIR": str(project_root),
            "DATADIFF_TMUX_SESSION": session_name,
            "DATADIFF_COMMAND": _long_running_command(),
            "DATADIFF_FINAL_READINESS_MANIFEST_INDEX": "reports/final-index.json",
            "DATADIFF_FINAL_READINESS_EXTRA_MANIFESTS": "reports/ledger-a.json,reports/ledger-b.json",
            "DATADIFF_FINAL_READINESS_FAIL_ON_MISSING": "1",
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
        payload = json.loads(freeze_manifest.read_text(encoding="utf-8"))

        assert status["final_readiness_manifest_index"] == "reports/final-index.json"
        assert status["final_readiness_extra_manifests"] == "reports/ledger-a.json,reports/ledger-b.json"
        assert status["final_readiness_fail_on_missing"] == "1"
        assert payload["post_run_readiness_config"] == {
            "manifest_index": "reports/final-index.json",
            "extra_manifests": "reports/ledger-a.json,reports/ledger-b.json",
            "fail_on_missing": True,
        }
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
            readiness_md="{project_root / 'reports' / 'final-readiness-finished.md'}"
            readiness_json="{project_root / 'reports' / 'final-readiness-finished.json'}"
            classify_dir="{project_root / 'reports' / 'classify-run-experiment-finished'}"
            printf 'summary\\n' > "${{summary_md}}"
            printf 'summary\\n' > "${{summary_csv}}"
            printf 'aggregate\\n' > "${{aggregate_csv}}"
            printf 'analysis\\n' > "${{analysis_md}}"
            printf 'analysis\\n' > "${{analysis_csv}}"
            printf 'methodology\\n' > "${{methodology_md}}"
            printf '{{"ok": true}}\\n' > "${{methodology_json}}"
            printf 'readiness\\n' > "${{readiness_md}}"
            printf '{{"ready": true}}\\n' > "${{readiness_json}}"
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
            echo "final_readiness_markdown=${{readiness_md}}"
            echo "final_readiness_json=${{readiness_json}}"
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
    assert Path(final_status["final_readiness_markdown"]).is_file()
    assert Path(final_status["final_readiness_json"]).is_file()
    assert Path(final_status["classify_run_dir"]).is_dir()
    assert final_status["classify_run_count"] == "1"
    assert (Path(final_status["classify_run_dir"]) / "run-a.json").is_file()
