#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLANCER_ROOT = Path(
    os.environ.get(
        "SQLANCER_ROOT",
        str(PROJECT_ROOT / "experiments" / "external_tools" / "sqlancer_duckdb153"),
    )
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "external-baselines"

SUITE_CONFIGS: dict[str, tuple[str, str, str]] = {
    "duckdb-query-partitioning": ("duckdb", "QUERY_PARTITIONING", "SQLancer DuckDB query partitioning"),
    "duckdb-norec": ("duckdb", "NOREC", "SQLancer DuckDB NoREC"),
    "sqlite3-norec": ("sqlite3", "NoREC", "SQLancer SQLite3 NoREC"),
    "sqlite3-query-partitioning": ("sqlite3", "QUERY_PARTITIONING", "SQLancer SQLite3 query partitioning"),
}


@dataclass(frozen=True, slots=True)
class SQLancerRunSpec:
    suite: str
    dbms: str
    oracle: str
    seed: int
    command: list[str]
    label: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["shell"] = shell_join(self.command)
        return data


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    runs = [run_spec_from_dict(run["spec"]) for run in manifest["runs"]]
    if bool(getattr(args, "execute", False)):
        executed_runs = []
        for index, spec in enumerate(runs):
            executed_runs.append(execute_run(spec, args=args, run_index=index))
        manifest["runs"] = executed_runs
        manifest["summary"] = summarize_runs(executed_runs)
    write_manifest(manifest, Path(str(args.output_manifest)))
    print(f"external baseline manifest: {args.output_manifest}")
    if not bool(getattr(args, "execute", False)):
        print("planned commands:")
        for run in manifest["runs"]:
            print(run["spec"]["shell"])
    return int(any(int(run.get("returncode", 0) or 0) != 0 for run in manifest["runs"]))


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    validate_log_options(args)
    sqlancer_root = Path(str(args.sqlancer_root)).expanduser()
    jar = resolve_jar(sqlancer_root, str(args.jar))
    target_versions = target_version_metadata(sqlancer_root)
    suites = selected_suites(args.suite)
    seeds = parse_int_list(str(args.seeds))
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer seed")
    run_specs = [
        build_run_spec(
            suite=suite,
            seed=seed,
            jar=jar,
            num_threads=int(args.num_threads),
            timeout_seconds=int(args.timeout_seconds),
            num_queries=int(args.num_queries),
            max_generated_databases=int(args.max_generated_databases),
            log_each_select=bool(args.log_each_select),
            log_execution_time=bool(args.log_execution_time),
        )
        for suite in suites
        for seed in seeds
    ]
    return {
        "schema_version": "external-sota-baseline-manifest-v1",
        "generated_at": utc_now_iso(),
        "tool": "sqlancer",
        "paper_role": "external_sota_baseline",
        "counts_as_datadiff_real_bugs": False,
        "counting_policy": (
            "SQLancer baseline findings must be triaged independently and must not be counted as "
            "DataDiffFuzz latest-version bug families."
        ),
        "sqlancer": {
            "root": str(sqlancer_root),
            "git_commit": git_commit(sqlancer_root),
            "jar": str(jar),
            "jar_exists": jar.is_file(),
            "jar_sha256": sha256_file(jar) if jar.is_file() else "",
            "java_version": java_version(),
        },
        "target_versions": target_versions,
        "strict_target_version_match": bool(target_versions.get("strict_duckdb_match")),
        "budget": {
            "num_threads": int(args.num_threads),
            "timeout_seconds": int(args.timeout_seconds),
            "num_queries": int(args.num_queries),
            "max_generated_databases": int(args.max_generated_databases),
            "process_timeout_seconds": int(args.process_timeout_seconds),
        },
        "suites": suites,
        "seeds": seeds,
        "runs": [{"spec": spec.to_dict(), "status": "planned"} for spec in run_specs],
        "summary": {
            "planned_run_count": len(run_specs),
            "executed_run_count": 0,
            "successful_run_count": 0,
        },
    }


def build_run_spec(
    *,
    suite: str,
    seed: int,
    jar: Path,
    num_threads: int,
    timeout_seconds: int,
    num_queries: int,
    max_generated_databases: int,
    log_each_select: bool,
    log_execution_time: bool,
) -> SQLancerRunSpec:
    dbms, oracle, label = SUITE_CONFIGS[suite]
    command = [
        "java",
        "-jar",
        str(jar),
        "--num-threads",
        str(num_threads),
        "--timeout-seconds",
        str(timeout_seconds),
        "--num-queries",
        str(num_queries),
        "--max-generated-databases",
        str(max_generated_databases),
        "--log-each-select",
        str_bool(log_each_select),
        "--log-execution-time",
        str_bool(log_execution_time),
        "--print-progress-summary",
        "true",
        "--random-seed",
        str(seed),
        dbms,
        "--oracle",
        oracle,
    ]
    return SQLancerRunSpec(
        suite=suite,
        dbms=dbms,
        oracle=oracle,
        seed=seed,
        command=command,
        label=f"{label} seed={seed}",
    )


def run_spec_from_dict(data: dict[str, object]) -> SQLancerRunSpec:
    return SQLancerRunSpec(
        suite=str(data["suite"]),
        dbms=str(data["dbms"]),
        oracle=str(data["oracle"]),
        seed=int(data["seed"]),
        command=[str(part) for part in data["command"]],
        label=str(data["label"]),
    )


def execute_run(spec: SQLancerRunSpec, *, args: argparse.Namespace, run_index: int) -> dict[str, object]:
    started = time.time()
    started_at = utc_now_iso()
    log_dir = Path(str(args.log_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"sqlancer-{spec.suite}-seed{spec.seed}-{run_index:03d}.log"
    timeout = int(args.process_timeout_seconds)
    status = "completed"
    timed_out = False
    try:
        proc = subprocess.Popen(
            spec.command,
            cwd=str(Path(str(args.sqlancer_root)).expanduser()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except OSError as exc:
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        proc = None
        returncode = 127
        status = "failed"
    if proc is not None:
        drain_thread = threading.Thread(
            target=_drain_process_output,
            args=(proc, log_path),
            daemon=True,
        )
        drain_thread.start()
        deadline = started + max(1, timeout)
        while proc.poll() is None:
            if time.time() >= deadline:
                timed_out = True
                status = "timeout"
                terminate_process_tree(proc)
                break
            time.sleep(0.5)
        try:
            returncode = int(proc.wait(timeout=15))
        except subprocess.TimeoutExpired:
            timed_out = True
            status = "timeout"
            kill_process_tree(proc)
            returncode = int(proc.wait(timeout=15))
        drain_thread.join(timeout=15)
        if timed_out:
            returncode = 124
    elapsed = time.time() - started
    stats = parse_sqlancer_summary_file(log_path)
    failure = parse_sqlancer_failure_signal_file(log_path, returncode=returncode)
    return {
        "spec": spec.to_dict(),
        "status": status if returncode == 0 else "failed" if status == "completed" else status,
        "returncode": returncode,
        "started_at": started_at,
        "completed_at": utc_now_iso(),
        "elapsed_seconds": round(elapsed, 3),
        "log_file": project_relative(log_path),
        "log_streaming": True,
        "stats": stats,
        "failure_signal": failure,
        "stdout_tail": tail_file_lines(log_path, 30),
    }


def _drain_process_output(proc: subprocess.Popen[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            handle.write(line)
            handle.flush()


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)


def kill_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            return


def parse_sqlancer_summary(output: str) -> dict[str, object]:
    stats: dict[str, object] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "Executed " in line and " queries" in line and "(" in line:
            stats["progress_line"] = line
            maybe = line.split("Executed ", 1)[-1].split(" queries", 1)[0]
            stats["last_progress_queries"] = parse_human_int(maybe)
        elif line.endswith("queries"):
            stats["summary_queries"] = parse_human_int(line[:-7].strip())
        elif line.endswith("databases"):
            stats["summary_databases"] = parse_human_int(line[:-9].strip())
        elif line.endswith("unsuccessfully-executed statements"):
            stats["summary_unsuccessful_statements"] = parse_human_int(
                line[: -len("unsuccessfully-executed statements")].strip()
            )
        elif line.endswith("successfully-executed statements"):
            stats["summary_successful_statements"] = parse_human_int(
                line[: -len("successfully-executed statements")].strip()
            )
    return stats


def parse_sqlancer_summary_file(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    stats: dict[str, object] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            stats.update(parse_sqlancer_summary(raw_line))
    return stats


def parse_sqlancer_failure_signal(output: str, *, returncode: int) -> dict[str, object]:
    indicators = [
        "java.lang.AssertionError",
        "java.lang.Exception",
        "java.lang.RuntimeException",
        "AssertionError",
        "potential bug",
        "found a bug",
        "database bug",
    ]
    lines = output.splitlines()
    matched_lines: list[str] = []
    lower_indicators = [item.lower() for item in indicators]
    for line in lines:
        lower = line.lower()
        if any(indicator in lower for indicator in lower_indicators):
            matched_lines.append(line.strip())
    return {
        "has_signal": bool(returncode != 0 or matched_lines),
        "returncode_nonzero": bool(returncode != 0),
        "matched_line_count": len(matched_lines),
        "matched_lines": matched_lines[:20],
    }


def parse_sqlancer_failure_signal_file(path: Path, *, returncode: int) -> dict[str, object]:
    if not path.is_file():
        return parse_sqlancer_failure_signal("", returncode=returncode)
    indicators = [
        "java.lang.AssertionError",
        "java.lang.Exception",
        "java.lang.RuntimeException",
        "AssertionError",
        "potential bug",
        "found a bug",
        "database bug",
    ]
    lower_indicators = [item.lower() for item in indicators]
    matched_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            lower = raw_line.lower()
            if any(indicator in lower for indicator in lower_indicators):
                matched_lines.append(raw_line.strip())
                if len(matched_lines) >= 20:
                    break
    return {
        "has_signal": bool(returncode != 0 or matched_lines),
        "returncode_nonzero": bool(returncode != 0),
        "matched_line_count": len(matched_lines),
        "matched_lines": matched_lines,
    }


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    total_queries = sum(int(run.get("stats", {}).get("summary_queries", 0) or 0) for run in runs)
    total_databases = sum(int(run.get("stats", {}).get("summary_databases", 0) or 0) for run in runs)
    return {
        "planned_run_count": len(runs),
        "executed_run_count": len(runs),
        "successful_run_count": sum(1 for run in runs if run_returncode(run) == 0),
        "failed_run_count": sum(1 for run in runs if run_returncode(run) != 0),
        "failure_signal_count": sum(
            1
            for run in runs
            if isinstance(run.get("failure_signal"), dict) and bool(run.get("failure_signal", {}).get("has_signal"))
        ),
        "total_summary_queries": total_queries,
        "total_summary_databases": total_databases,
    }


def run_returncode(run: dict[str, object]) -> int:
    value = run.get("returncode", 1)
    return 1 if value is None else int(value)


def validate_log_options(args: argparse.Namespace) -> None:
    if bool(args.log_each_select):
        return
    raise SystemExit(
        "Current SQLancer mainline cannot run with --no-log-each-select: "
        "MainOptions.logExecutionTime() asserts when log-each-select is false. "
        "Keep SQLancer's statement logging enabled for fair baseline runs; "
        "use --no-log-execution-time only to reduce timing-log overhead."
    )


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SQLancer as an external SOTA baseline and emit a reproducible manifest."
    )
    parser.add_argument("--sqlancer-root", default=str(DEFAULT_SQLANCER_ROOT))
    parser.add_argument("--jar", default="", help="SQLancer jar path; defaults to <sqlancer-root>/target/sqlancer-2.0.0.jar")
    parser.add_argument(
        "--suite",
        action="append",
        choices=sorted(SUITE_CONFIGS),
        default=[],
        help="SQLancer DBMS/oracle suite to run; may be repeated",
    )
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--num-queries", type=int, default=100000)
    parser.add_argument("--max-generated-databases", type=int, default=20)
    parser.add_argument("--process-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--log-each-select",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "forward SQLancer --log-each-select; current SQLancer mainline requires this to stay "
            "enabled because MainOptions.logExecutionTime() asserts when it is disabled"
        ),
    )
    parser.add_argument(
        "--log-execution-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forward SQLancer --log-execution-time; disable to reduce timing-log overhead",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--output-manifest",
        default=str(DEFAULT_OUTPUT_DIR / f"sqlancer-baseline-{utc_timestamp()}.json"),
    )
    parser.add_argument(
        "--log-dir",
        default=str(DEFAULT_OUTPUT_DIR / "logs"),
    )
    return parser.parse_args()


def selected_suites(values: Iterable[str]) -> list[str]:
    suites = [str(value).strip() for value in values if str(value).strip()]
    return suites or ["duckdb-query-partitioning"]


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        result.append(int(text))
    return result


def resolve_jar(sqlancer_root: Path, value: str) -> Path:
    if value:
        return Path(value).expanduser()
    return sqlancer_root / "target" / "sqlancer-2.0.0.jar"


def git_commit(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(path),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def java_version() -> str:
    try:
        proc = subprocess.run(
            ["java", "-version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError:
        return ""
    return " ".join(line.strip() for line in (proc.stdout or "").splitlines() if line.strip())


def target_version_metadata(sqlancer_root: Path) -> dict[str, object]:
    datadiff = datadiff_duckdb_info()
    sqlancer = {
        "duckdb_jdbc_version": sqlancer_duckdb_jdbc_version(sqlancer_root),
    }
    datadiff_normalized = normalize_duckdb_version(str(datadiff.get("engine_version", "")))
    sqlancer_normalized = normalize_duckdb_version(str(sqlancer.get("duckdb_jdbc_version", "")))
    strict_match = bool(datadiff_normalized and sqlancer_normalized and datadiff_normalized == sqlancer_normalized)
    return {
        "policy": (
            "Final strict DuckDB/SQLancer head-to-head comparisons must use the same DuckDB "
            "engine version. Version-mismatched runs are pipeline/support pilots only."
        ),
        "datadiff_duckdb": datadiff,
        "sqlancer_duckdb": sqlancer,
        "normalized": {
            "datadiff_duckdb": datadiff_normalized,
            "sqlancer_duckdb_jdbc": sqlancer_normalized,
        },
        "strict_duckdb_match": strict_match,
    }


def datadiff_duckdb_info() -> dict[str, object]:
    python_path = PROJECT_ROOT / ".venv" / "bin" / "python"
    if not python_path.is_file():
        return {"python": str(python_path), "available": False, "error": "missing .venv python"}
    script = (
        "import json, duckdb\n"
        "info={'available': True, 'python_package_version': getattr(duckdb, '__version__', '')}\n"
        "try:\n"
        "    row=duckdb.sql('pragma version').fetchone()\n"
        "    info['engine_version']=str(row[0]) if row else ''\n"
        "    info['engine_commit']=str(row[1]) if row and len(row)>1 else ''\n"
        "    info['engine_codename']=str(row[2]) if row and len(row)>2 else ''\n"
        "except Exception as exc:\n"
        "    info['error']=type(exc).__name__ + ': ' + str(exc)\n"
        "print(json.dumps(info, sort_keys=True))\n"
    )
    try:
        proc = subprocess.run(
            [str(python_path), "-c", script],
            cwd=str(PROJECT_ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except Exception as exc:
        return {"python": str(python_path), "available": False, "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {
            "python": str(python_path),
            "available": False,
            "error": (proc.stderr or proc.stdout or "").strip(),
        }
    try:
        data = json.loads((proc.stdout or "").strip())
    except json.JSONDecodeError:
        return {"python": str(python_path), "available": False, "error": "unparseable duckdb version output"}
    if isinstance(data, dict):
        data["python"] = str(python_path)
        return data
    return {"python": str(python_path), "available": False, "error": "unexpected duckdb version payload"}


def sqlancer_duckdb_jdbc_version(sqlancer_root: Path) -> str:
    pom = sqlancer_root / "pom.xml"
    if not pom.is_file():
        return ""
    try:
        root = ET.parse(pom).getroot()
    except ET.ParseError:
        return ""
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}", 1)[0].strip("{")
    prefix = f"{{{namespace}}}" if namespace else ""
    for dependency in root.findall(f".//{prefix}dependency"):
        group_id = dependency.findtext(f"{prefix}groupId", default="")
        artifact_id = dependency.findtext(f"{prefix}artifactId", default="")
        if group_id == "org.duckdb" and artifact_id == "duckdb_jdbc":
            return dependency.findtext(f"{prefix}version", default="").strip()
    return ""


def normalize_duckdb_version(value: str) -> str:
    match = re_search_version(value)
    if not match:
        return ""
    parts = match.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:3])
    return match


def re_search_version(value: str) -> str:
    import re

    match = re.search(r"(\d+(?:\.\d+){1,3})", value)
    return match.group(1) if match else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_human_int(value: str) -> int:
    text = value.strip().replace(",", "")
    if not text:
        return 0
    multiplier = 1
    suffix = text[-1].lower()
    if suffix == "k":
        multiplier = 1_000
        text = text[:-1]
    elif suffix == "m":
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return int(float(text.strip()) * multiplier)
    except ValueError:
        return 0


def tail_lines(value: str, count: int) -> list[str]:
    return value.splitlines()[-count:]


def tail_file_lines(path: Path, count: int) -> list[str]:
    if not path.is_file() or count <= 0:
        return []
    lines: deque[str] = deque(maxlen=count)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lines.append(line.rstrip("\n"))
    return list(lines)


def shell_join(command: list[str]) -> str:
    return " ".join(sh_quote(part) for part in command)


def sh_quote(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%"
    if all(char in safe for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def str_bool(value: bool) -> str:
    return "true" if value else "false"


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
