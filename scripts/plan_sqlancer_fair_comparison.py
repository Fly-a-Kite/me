#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SQLANCER_ROOT = PROJECT_ROOT / "experiments" / "external_tools" / "sqlancer_duckdb153"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "external-baselines" / "plans"


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    seeds = parse_int_list(str(args.seeds))
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    run_id = str(args.run_id or f"sqlancer-fair-{utc_timestamp()}")
    output_dir = Path(str(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{run_id}.sh"
    manifest_path = output_dir / f"{run_id}.json"
    commands = build_commands(args, seeds=seeds, run_id=run_id)
    write_script(script_path, commands)
    write_manifest(manifest_path, args=args, seeds=seeds, run_id=run_id, commands=commands)
    print(f"fair comparison plan: {manifest_path}")
    print(f"fair comparison script: {script_path}")
    if bool(args.print_commands):
        for command in commands:
            print(command["shell"])
    return 0


def build_commands(args: argparse.Namespace, *, seeds: list[int], run_id: str) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    duration_seconds = int(args.duration_seconds)
    process_timeout_seconds = duration_seconds + int(args.timeout_slack_seconds)
    for suite in string_list(args.sqlancer_suite):
        commands.append(
            {
                "tool": "sqlancer",
                "suite": suite,
                "shell": shell_join(
                    [
                        "rtk",
                        "proxy",
                        "python",
                        "scripts/run_sqlancer_baseline.py",
                        "--sqlancer-root",
                        str(args.sqlancer_root),
                        "--suite",
                        suite,
                        "--seeds",
                        ",".join(str(seed) for seed in seeds),
                        "--num-threads",
                        "1",
                        "--timeout-seconds",
                        str(duration_seconds),
                        "--num-queries",
                        str(args.sqlancer_num_queries),
                        "--max-generated-databases",
                        str(args.sqlancer_max_generated_databases),
                        "--process-timeout-seconds",
                        str(process_timeout_seconds),
                        "--no-log-execution-time",
                        "--execute",
                        "--output-manifest",
                        str(
                            Path(args.sqlancer_manifest_dir)
                            / f"{run_id}-{suite}-{duration_seconds}s.json"
                        ),
                    ]
                ),
            }
        )
    for seed in seeds:
        datadiff_experiment_meta = {
            "analysis_tags": ["comparison", "external_baseline", "sqlancer", "scope_limited"],
            "comparison_group": "external_sqlancer_scope_limited",
            "comparison_role": "datadiff_under_test",
            "counts_as_real_bugs": False,
            "external_baseline": "sqlancer",
            "matrix_id": "external_sota_sqlancer_scope_comparison",
            "matrix_title": "External SQLancer Scope-Limited Comparison",
            "purpose": "Scope-limited support comparison against SQLancer under equal wall-clock, seed, thread, and target-version policy.",
            "rq_tags": ["RQ4", "RQ6"],
            "scope_by_target_suite": {str(args.datadiff_target_suite): "sql_oriented"},
            "scope_kind": "sql_oriented",
            "target_suites": [str(args.datadiff_target_suite)],
            "track": "comparison",
        }
        commands.append(
            {
                "tool": "datadiff",
                "suite": str(args.datadiff_target_suite),
                "seed": seed,
                "shell": shell_join(
                    [
                        "rtk",
                        ".venv/bin/datadiff",
                        "experiment",
                        "--duration",
                        f"{duration_seconds}s",
                        "--seeds",
                        str(seed),
                        "--presets",
                        str(args.datadiff_preset),
                        "--target-suite",
                        str(args.datadiff_target_suite),
                        "--evidence-mode",
                        "comparison",
                        "--schedule",
                        "adaptive",
                        "--batch-duration",
                        str(args.datadiff_batch_duration),
                        "--adaptive-learning-weight",
                        str(args.datadiff_adaptive_learning_weight),
                        "--scheduler-annealing-temperature",
                        str(args.datadiff_scheduler_annealing_temperature),
                        "--scheduler-annealing-decay",
                        str(args.datadiff_scheduler_annealing_decay),
                        "--scheduler-annealing-min-temperature",
                        str(args.datadiff_scheduler_annealing_min_temperature),
                        "--run-theme",
                        f"sqlancer-fair-comparison:{args.datadiff_target_suite}:{args.datadiff_preset}:seed{seed}",
                        "--paper-notes",
                        "Scope-limited SQLancer comparison pilot; support evidence only.",
                        "--artifact-limit",
                        str(args.datadiff_artifact_limit),
                        "--log-level",
                        str(args.datadiff_log_level),
                        "--jobs",
                        "1",
                        "--persist-closed-loop-state",
                        "--skip-run-reports",
                        "--skip-paper-journal",
                        "--experiment-meta",
                        json.dumps(datadiff_experiment_meta, sort_keys=True),
                    ]
                ),
            }
        )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a fair SQLancer-vs-DataDiffFuzz scope-limited comparison."
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--duration-seconds", type=int, default=7200)
    parser.add_argument("--timeout-slack-seconds", type=int, default=300)
    parser.add_argument("--seeds", default="1,1001,2001")
    parser.add_argument("--sqlancer-root", default=str(DEFAULT_SQLANCER_ROOT))
    parser.add_argument(
        "--sqlancer-suite",
        action="append",
        default=["duckdb-query-partitioning", "duckdb-norec"],
    )
    parser.add_argument("--sqlancer-num-queries", type=int, default=1000000000)
    parser.add_argument("--sqlancer-max-generated-databases", type=int, default=1000000)
    parser.add_argument(
        "--sqlancer-manifest-dir",
        default=str(PROJECT_ROOT / "reports" / "external-baselines"),
    )
    parser.add_argument("--datadiff-target-suite", default="embedded_sql")
    parser.add_argument("--datadiff-preset", default="live_duckdb_issue_focus")
    parser.add_argument("--datadiff-batch-duration", default="10m")
    parser.add_argument("--datadiff-adaptive-learning-weight", type=float, default=0.75)
    parser.add_argument("--datadiff-scheduler-annealing-temperature", type=float, default=0.35)
    parser.add_argument("--datadiff-scheduler-annealing-decay", type=float, default=0.985)
    parser.add_argument("--datadiff-scheduler-annealing-min-temperature", type=float, default=0.02)
    parser.add_argument("--datadiff-artifact-limit", type=int, default=50)
    parser.add_argument("--datadiff-log-level", choices=["full", "compact", "minimal"], default="minimal")
    parser.add_argument("--print-commands", action="store_true")
    return parser.parse_args()


def write_script(path: Path, commands: list[dict[str, object]]) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd " + sh_quote(str(PROJECT_ROOT)),
        "",
    ]
    for index, command in enumerate(commands, start=1):
        lines.append(f"echo '[{index}/{len(commands)}] {command['tool']} {command.get('suite', '')} {command.get('seed', '')}'")
        lines.append(str(command["shell"]))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    seeds: list[int],
    run_id: str,
    commands: list[dict[str, object]],
) -> None:
    payload = {
        "schema_version": "sqlancer-fair-comparison-plan-v1",
        "generated_at": utc_now_iso(),
        "run_id": run_id,
        "purpose": "Scope-limited DuckDB/SQL fair comparison between SQLancer and DataDiffFuzz.",
        "counting_policy": "Support evidence only; count only independently confirmed unique bug families.",
        "duration_seconds_per_run": int(args.duration_seconds),
        "seeds": seeds,
        "sqlancer_suites": string_list(args.sqlancer_suite),
        "datadiff_target_suite": str(args.datadiff_target_suite),
        "datadiff_preset": str(args.datadiff_preset),
        "commands": commands,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if text:
            result.append(int(text))
    return result


def string_list(values: list[str] | tuple[str, ...] | str) -> list[str]:
    if isinstance(values, str):
        values = [values]
    result: list[str] = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                result.append(text)
    return result


def shell_join(command: list[object]) -> str:
    return " ".join(sh_quote(part) for part in command)


def sh_quote(value: object) -> str:
    text = str(value)
    if not text:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_+-=.,/:@%"
    if all(char in safe for char in text):
        return text
    return "'" + text.replace("'", "'\"'\"'") + "'"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
