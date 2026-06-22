from __future__ import annotations

import importlib.util
import textwrap
import sys
import time
from argparse import Namespace
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_sqlancer_baseline.py"
    spec = importlib.util.spec_from_file_location("run_sqlancer_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path, **overrides):
    root = tmp_path / "sqlancer"
    jar = root / "target" / "sqlancer-2.0.0.jar"
    jar.parent.mkdir(parents=True)
    jar.write_text("fake jar", encoding="utf-8")
    data = {
        "sqlancer_root": str(root),
        "jar": str(jar),
        "suite": [],
        "seeds": "1,1001",
        "num_threads": 1,
        "timeout_seconds": 8,
        "num_queries": 25,
        "max_generated_databases": 1,
        "process_timeout_seconds": 20,
        "log_each_select": True,
        "log_execution_time": True,
        "execute": False,
        "output_manifest": str(tmp_path / "manifest.json"),
        "log_dir": str(tmp_path / "logs"),
    }
    data.update(overrides)
    return Namespace(**data)


def test_default_plan_targets_duckdb_query_partitioning(tmp_path):
    module = _module()
    module.datadiff_duckdb_info = lambda: {
        "available": True,
        "python_package_version": "1.5.3",
        "engine_version": "v1.5.3",
        "engine_commit": "abc",
    }
    (tmp_path / "sqlancer").mkdir()
    (tmp_path / "sqlancer" / "pom.xml").write_text(
        textwrap.dedent(
            """\
            <project xmlns="http://maven.apache.org/POM/4.0.0">
              <dependencies>
                <dependency>
                  <groupId>org.duckdb</groupId>
                  <artifactId>duckdb_jdbc</artifactId>
                  <version>1.5.3.0</version>
                </dependency>
              </dependencies>
            </project>
            """
        ),
        encoding="utf-8",
    )

    manifest = module.build_manifest(_args(tmp_path))

    assert manifest["schema_version"] == "external-sota-baseline-manifest-v1"
    assert manifest["tool"] == "sqlancer"
    assert manifest["counts_as_datadiff_real_bugs"] is False
    assert manifest["suites"] == ["duckdb-query-partitioning"]
    assert manifest["seeds"] == [1, 1001]
    assert len(manifest["runs"]) == 2
    first = manifest["runs"][0]["spec"]
    assert first["dbms"] == "duckdb"
    assert first["oracle"] == "QUERY_PARTITIONING"
    assert "--random-seed" in first["command"]
    assert "duckdb" in first["command"]
    assert first["command"][-2:] == ["--oracle", "QUERY_PARTITIONING"]
    assert manifest["summary"]["planned_run_count"] == 2
    assert manifest["target_versions"]["normalized"] == {
        "datadiff_duckdb": "1.5.3",
        "sqlancer_duckdb_jdbc": "1.5.3",
    }
    assert manifest["strict_target_version_match"] is True


def test_target_version_metadata_marks_duckdb_mismatch(tmp_path):
    module = _module()
    module.datadiff_duckdb_info = lambda: {
        "available": True,
        "python_package_version": "1.5.3",
        "engine_version": "v1.5.3",
    }
    sqlancer_root = tmp_path / "sqlancer"
    sqlancer_root.mkdir()
    (sqlancer_root / "pom.xml").write_text(
        textwrap.dedent(
            """\
            <project xmlns="http://maven.apache.org/POM/4.0.0">
              <dependencies>
                <dependency>
                  <groupId>org.duckdb</groupId>
                  <artifactId>duckdb_jdbc</artifactId>
                  <version>1.3.0.0</version>
                </dependency>
              </dependencies>
            </project>
            """
        ),
        encoding="utf-8",
    )

    metadata = module.target_version_metadata(sqlancer_root)

    assert metadata["sqlancer_duckdb"]["duckdb_jdbc_version"] == "1.3.0.0"
    assert metadata["normalized"]["datadiff_duckdb"] == "1.5.3"
    assert metadata["normalized"]["sqlancer_duckdb_jdbc"] == "1.3.0"
    assert metadata["strict_duckdb_match"] is False


def test_log_controls_are_forwarded_to_sqlancer(tmp_path):
    module = _module()

    manifest = module.build_manifest(
        _args(tmp_path, log_execution_time=False)
    )

    command = manifest["runs"][0]["spec"]["command"]
    assert command[command.index("--log-each-select") + 1] == "true"
    assert command[command.index("--log-execution-time") + 1] == "false"


def test_no_log_each_select_is_rejected_for_current_sqlancer_mainline(tmp_path):
    module = _module()

    try:
        module.build_manifest(_args(tmp_path, log_each_select=False))
    except SystemExit as exc:
        assert "cannot run with --no-log-each-select" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_explicit_suites_and_seed_list_are_cross_product(tmp_path):
    module = _module()

    manifest = module.build_manifest(
        _args(tmp_path, suite=["duckdb-norec", "sqlite3-norec"], seeds="7")
    )

    specs = [run["spec"] for run in manifest["runs"]]
    assert [(spec["dbms"], spec["oracle"], spec["seed"]) for spec in specs] == [
        ("duckdb", "NOREC", 7),
        ("sqlite3", "NoREC", 7),
    ]


def test_parse_sqlancer_summary_handles_human_counts():
    module = _module()
    output = """
Overall execution statistics
============================
        2k queries
        94 databases
       61k successfully-executed statements
       43k unsuccessfully-executed statements
"""

    stats = module.parse_sqlancer_summary(output)

    assert stats["summary_queries"] == 2000
    assert stats["summary_databases"] == 94
    assert stats["summary_successful_statements"] == 61000
    assert stats["summary_unsuccessful_statements"] == 43000


def test_parse_sqlancer_failure_signal_tracks_java_or_bug_indicators_only():
    module = _module()

    normal_duckdb_error = module.parse_sqlancer_failure_signal(
        "duckdb.duckdbexception: parser error near generated input",
        returncode=0,
    )
    assertion = module.parse_sqlancer_failure_signal(
        "--java.lang.AssertionError\n--\tat sqlancer.MainOptions.logExecutionTime",
        returncode=0,
    )

    assert normal_duckdb_error["has_signal"] is False
    assert assertion["has_signal"] is True
    assert assertion["matched_line_count"] == 1


def test_file_parsers_support_streamed_sqlancer_logs(tmp_path):
    module = _module()
    log = tmp_path / "sqlancer.log"
    log.write_text(
        textwrap.dedent(
            """\
            progress
                    2k queries
                    94 databases
            java.lang.AssertionError: possible oracle failure
            """
        ),
        encoding="utf-8",
    )

    stats = module.parse_sqlancer_summary_file(log)
    failure = module.parse_sqlancer_failure_signal_file(log, returncode=0)

    assert stats["summary_queries"] == 2000
    assert stats["summary_databases"] == 94
    assert failure["has_signal"] is True
    assert failure["matched_line_count"] == 1
    assert module.tail_file_lines(log, 2) == [
        "        94 databases",
        "java.lang.AssertionError: possible oracle failure",
    ]


def test_execute_run_streams_log_and_times_out_process_tree(tmp_path):
    module = _module()
    script = tmp_path / "fake_sqlancer.py"
    script.write_text(
        textwrap.dedent(
            """\
            import sys, time
            print('1 queries', flush=True)
            time.sleep(5)
            """
        ),
        encoding="utf-8",
    )
    spec = module.SQLancerRunSpec(
        suite="duckdb-query-partitioning",
        dbms="duckdb",
        oracle="QUERY_PARTITIONING",
        seed=1,
        command=[sys.executable, str(script)],
        label="fake sqlancer",
    )

    started = time.time()
    run = module.execute_run(
        spec,
        args=_args(
            tmp_path,
            process_timeout_seconds=1,
            log_dir=str(tmp_path / "logs"),
        ),
        run_index=0,
    )

    assert time.time() - started < 4
    assert run["status"] == "timeout"
    assert run["returncode"] == 124
    assert run["log_streaming"] is True
    assert run["stdout_tail"] == ["1 queries"]
    assert Path(tmp_path / run["log_file"]).is_file() or (tmp_path / "logs" / "sqlancer-duckdb-query-partitioning-seed1-000.log").is_file()


def test_summarize_runs_counts_zero_returncode_as_success():
    module = _module()

    summary = module.summarize_runs(
        [
            {
                "returncode": 0,
                "failure_signal": {"has_signal": True},
                "stats": {
                    "summary_queries": 171,
                    "summary_databases": 12,
                },
            }
        ]
    )

    assert summary["successful_run_count"] == 1
    assert summary["failed_run_count"] == 0
    assert summary["failure_signal_count"] == 1
    assert summary["total_summary_queries"] == 171
