#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES  # noqa: E402
from datadiff.historical import list_historical_bugs  # noqa: E402
from datadiff.util import REPORTS_DIR, utc_now  # noqa: E402

DATADIFF = Path(sys.prefix) / "bin" / "datadiff"
if not DATADIFF.exists():
    DATADIFF = PROJECT_ROOT / ".venv" / "bin" / "datadiff"


@dataclass(frozen=True, slots=True)
class FinalCommand:
    track: str
    name: str
    command: list[str]
    purpose: str
    count_as_real_bugs: bool
    expected_output: str
    notes: str = ""
    replay_bug_policy: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shell"] = shell_join(self.command)
        return data


LIVE_DISCOVERY_SUITES: tuple[tuple[str, str, str], ...] = (
    (
        "datafusion_cross",
        "live_datafusion",
        "Latest-version DataFusion differential discovery against pandas and DuckDB references.",
    ),
    (
        "dataframe_lazy",
        "live_polars_lazy",
        "Latest-version Polars eager/lazy consistency discovery.",
    ),
    (
        "arrow_cross",
        "live_arrow",
        "Latest-version Arrow/PyArrow cross-family discovery.",
    ),
    (
        "embedded_sql",
        "live_embedded_sql",
        "Latest-version embedded SQL cross-engine discovery.",
    ),
    (
        "latest_all_engines",
        "live_cross_family",
        "Broad latest-version cross-family discovery over every implemented real backend.",
    ),
)


def main() -> int:
    return main_with_args_for_test(parse_args())


def main_with_args_for_test(args: argparse.Namespace) -> int:
    if args.execute and args.track == "all":
        print(
            "refusing --track all --execute because live/latest and historical/vulnerable "
            "runs require different Python environments; execute track-specific plans from "
            "the matching venv instead.",
            file=sys.stderr,
        )
        return 2
    commands = build_plan(args)
    plan_path = write_plan(commands, args)
    print(f"final experiment plan: {plan_path}")
    for item in commands:
        print()
        print(f"[{item.track}] {item.name}")
        print(f"purpose: {item.purpose}")
        print(f"counts_as_real_bugs: {str(item.count_as_real_bugs).lower()}")
        if item.notes:
            print(f"notes: {item.notes}")
        print(shell_join(item.command))
    if args.execute:
        for item in commands:
            print(f"\nexecuting [{item.track}] {item.name}", flush=True)
            subprocess.run(item.command, cwd=PROJECT_ROOT, check=True)
    return 0


def jobs_arg(value: object) -> str:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    return str(max(1, int(text)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen final experiment command set. By default this "
            "only writes/prints a plan; pass --execute to run it."
        ),
    )
    parser.add_argument(
        "--track",
        choices=["all", "live", "historical", "seeded"],
        default="all",
        help="experiment track to plan",
    )
    parser.add_argument("--duration", default="24h", help="per-run wall-clock budget for live discovery")
    parser.add_argument("--live-seeds", default="1,1001,2001", help="comma-separated live discovery seeds")
    parser.add_argument("--historical-seeds", default=None, help="override historical replay seeds")
    parser.add_argument("--seeded-cases", type=int, default=5000, help="cases per seeded sensitivity run")
    parser.add_argument("--seeded-seeds", default="1,1001,2001,3001,4001", help="seeded sensitivity seeds")
    parser.add_argument("--jobs", default="auto", help="parallel experiment jobs, or 'auto'")
    parser.add_argument("--artifact-limit", type=int, default=50, help="bug artifacts per run")
    parser.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="minimal",
        help="run JSONL detail level; minimal is the default for 24h runs to reduce per-case IO",
    )
    parser.add_argument("--include-pending-historical", action="store_true")
    parser.add_argument("--skip-run-reports", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="execute commands instead of only printing them")
    return parser.parse_args()


def build_plan(args: argparse.Namespace) -> list[FinalCommand]:
    commands: list[FinalCommand] = []
    tracks = {"live", "historical", "seeded"} if args.track == "all" else {args.track}
    if "live" in tracks:
        commands.extend(live_discovery_commands(args))
    if "historical" in tracks:
        commands.extend(historical_replay_commands(args))
    if "seeded" in tracks:
        commands.append(seeded_sensitivity_command(args))
    return commands


def live_discovery_commands(args: argparse.Namespace) -> list[FinalCommand]:
    commands = []
    replay_sources = replay_source_issues()
    for suite, preset, purpose in LIVE_DISCOVERY_SUITES:
        cmd = [
            str(DATADIFF),
            "experiment",
            "--duration",
            str(args.duration),
            "--seeds",
            str(args.live_seeds),
            "--presets",
            preset,
            "--target-suite",
            suite,
            "--evidence-mode",
            "live",
            "--run-theme",
            f"final-live:{suite}:{preset}",
            "--paper-notes",
            purpose,
            "--replay-bug-source-issues",
            ",".join(replay_sources),
            "--artifact-limit",
            str(max(0, int(args.artifact_limit))),
            "--log-level",
            str(args.log_level),
            "--jobs",
            jobs_arg(args.jobs),
            "--skip-paper-journal",
        ]
        if args.skip_run_reports:
            cmd.append("--skip-run-reports")
        commands.append(
            FinalCommand(
                track="live",
                name=f"{suite}:{preset}",
                command=cmd,
                purpose=purpose,
                count_as_real_bugs=True,
                expected_output="runs/experiment-*.json plus reports from experiment-summary/analyze-experiment",
                notes=(
                    "Fresh/latest mode keeps enable_replay_bug=false and filters known replay probes. "
                    "Run without changing generator/oracle code after inspecting findings. "
                    "Count unique candidate bug families, then separately mark maintainer-confirmed bugs."
                ),
                replay_bug_policy={
                    "enable_replay_bug": False,
                    "source_issues": replay_sources,
                },
            )
        )
    return commands


def historical_replay_commands(args: argparse.Namespace) -> list[FinalCommand]:
    specs = list_historical_bugs(include_pending=bool(args.include_pending_historical))
    commands: list[FinalCommand] = []
    for spec in specs:
        if spec.replay_kind == "fixture":
            commands.append(_historical_fixture_replay_command(spec, args))
            continue
        seeds = args.historical_seeds or ",".join(str(seed) for seed in spec.default_seeds)
        artifact_limit = (
            args.artifact_limit
            if getattr(spec, "default_artifact_limit", None) is None
            else int(getattr(spec, "default_artifact_limit"))
        )
        log_level = str(getattr(spec, "default_log_level", "") or args.log_level)
        replay_sources = replay_source_issues(spec.issue_url)
        cmd = [
            str(DATADIFF),
            "experiment",
            "--cases",
            str(spec.default_cases),
            "--seeds",
            seeds,
            "--presets",
            ",".join(spec.default_presets),
            "--target-suite",
            spec.target_suite,
            "--evidence-mode",
            "historical",
            "--known-bug-id",
            spec.bug_id,
            "--target-version",
            spec.target_version,
            "--enable-replay-bug",
            "--replay-bug-source-issues",
            ",".join(replay_sources),
            "--run-theme",
            f"final-historical:{spec.bug_id}",
            "--paper-notes",
            f"Historical replay for {spec.bug_id}; status={spec.status}.",
            "--artifact-limit",
            str(max(0, int(artifact_limit))),
            "--log-level",
            log_level,
            "--jobs",
            jobs_arg(args.jobs),
            "--skip-paper-journal",
        ]
        if args.skip_run_reports:
            cmd.append("--skip-run-reports")
        commands.append(
            FinalCommand(
                track="historical",
                name=spec.bug_id,
                command=cmd,
                purpose=f"Replay previously reported {spec.project} bug on a vulnerable target version.",
                count_as_real_bugs=spec.status == "confirmed_fixed",
                expected_output="historical experiment manifest and expected-root detection metrics",
                notes=(
                    f"status={spec.status}; run inside an environment whose backend version is "
                    f"{spec.target_version}. Replay mode sets enable_replay_bug=true but uses the "
                    f"same generator/oracle/runner path as fresh mode. {spec.notes}"
                ).strip(),
                replay_bug_policy={
                    "enable_replay_bug": True,
                    "source_issues": replay_sources,
                },
            )
        )
    return commands


def _historical_fixture_replay_command(spec: object, args: argparse.Namespace) -> FinalCommand:
    replay_sources = replay_source_issues(str(getattr(spec, "issue_url", "")))
    cmd = [
        str(DATADIFF),
        "replay-fixture",
        "--spec",
        str(getattr(spec, "fixture_spec")),
        "--fixture-env",
        str(getattr(spec, "fixture_env")),
        "--target-suite",
        str(getattr(spec, "target_suite")),
        "--evidence-mode",
        "historical",
        "--known-bug-id",
        str(getattr(spec, "bug_id")),
        "--target-version",
        str(getattr(spec, "target_version")),
        "--run-theme",
        f"final-historical:{getattr(spec, 'bug_id')}",
        "--paper-notes",
        f"Historical fixture replay for {getattr(spec, 'bug_id')}; status={getattr(spec, 'status')}.",
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
    ]
    return FinalCommand(
        track="historical",
        name=str(getattr(spec, "bug_id")),
        command=cmd,
        purpose=f"Replay previously reported {getattr(spec, 'project')} bug using an upstream fixture.",
        count_as_real_bugs=getattr(spec, "status") == "confirmed_fixed",
        expected_output="runs/run-fixture-*.jsonl plus paper run journal entry",
        notes=(
            f"status={getattr(spec, 'status')}; set {getattr(spec, 'fixture_env')} to the "
            f"external fixture path before execution. {getattr(spec, 'notes')}"
        ).strip(),
        replay_bug_policy={
            "enable_replay_bug": True,
            "source_issues": replay_sources,
        },
    )


def seeded_sensitivity_command(args: argparse.Namespace) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.seeded_cases))),
        "--seeds",
        str(args.seeded_seeds),
        "--presets",
        "baseline,guided_filter,guided_groupby,guided_join,guided_mutate",
        "--target-suites",
        "seeded_filter,seeded_groupby,seeded_join,seeded_mutate",
        "--evidence-mode",
        "seeded",
        "--run-theme",
        "final-seeded-sensitivity",
        "--paper-notes",
        "Controlled injected-fault sensitivity run; not counted as real backend bugs.",
        "--artifact-limit",
        "1",
        "--log-level",
        "minimal",
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-paper-journal",
        "--skip-run-reports",
    ]
    return FinalCommand(
        track="seeded",
        name="seeded_sensitivity",
        command=cmd,
        purpose="Measure detection sensitivity and time-to-first on controlled injected faults.",
        count_as_real_bugs=False,
        expected_output="seeded-sensitivity report; do not include these in real bug counts",
        notes="Use this to support method validity, not as backend bug evidence.",
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
    )


def write_plan(commands: Iterable[FinalCommand], args: argparse.Namespace) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"final-experiment-plan-{utc_now().replace(':', '').replace('-', '').replace('Z', '')}-{time.time_ns()}.json"
    payload = {
        "created_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "datadiff": str(DATADIFF),
        "args": vars(args),
        "commands": [command.to_dict() for command in commands],
        "policy": {
            "freeze_rule": "Do not change generator, oracle, normalizer, or triage code after starting final runs.",
            "live_counts": "Live latest-version runs count candidate/confirmed real backend bugs after family deduplication.",
            "historical_counts": "Historical runs count only confirmed_fixed specs; pending/candidate specs are case studies.",
            "seeded_counts": "Seeded runs measure sensitivity only and do not count as real bugs.",
            "family_key": "root_cause + suspicious_backends",
            "replay_bug_gate": (
                "Final live commands explicitly keep enable_replay_bug=false; historical commands "
                "explicitly enable replay while using the same middle/bottom harness."
            ),
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def replay_source_issues(*extra_sources: str) -> list[str]:
    sources = [str(source).strip().rstrip("/") for source in DEFAULT_REPLAY_BUG_SOURCE_ISSUES]
    sources.extend(str(source).strip().rstrip("/") for source in extra_sources)
    return sorted({source for source in sources if source})


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
