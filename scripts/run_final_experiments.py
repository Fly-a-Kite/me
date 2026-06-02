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
from datadiff.experiment_catalog import (  # noqa: E402
    FINAL_COMPARISON_MATRIX,
    FINAL_LIVE_DISCOVERY_MATRIX,
    FINAL_MODULE_ABLATION_MATRIX,
    FINAL_PROTOCOL_TRACKS,
    FINAL_SEEDED_SENSITIVITY_MATRIX,
    FINAL_VALIDATION_MATRIX,
    build_historical_experiment_meta,
)
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
    experiment_meta: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["shell"] = shell_join(self.command)
        return data


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
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


def main_with_args_for_test(args: argparse.Namespace) -> int:
    # Compatibility alias for older tests and external wrappers.
    return run_with_args(args)


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
        choices=["all", *FINAL_PROTOCOL_TRACKS],
        default="all",
        help="experiment track to plan",
    )
    parser.add_argument("--duration", default="24h", help="per-run wall-clock budget for live discovery")
    parser.add_argument("--validation-cases", type=int, default=200, help="cases per short validation run")
    parser.add_argument("--validation-seeds", default="1,101", help="short validation seeds")
    parser.add_argument("--live-seeds", default="1,1001,2001", help="comma-separated live discovery seeds")
    parser.add_argument("--historical-seeds", default=None, help="override historical replay seeds")
    parser.add_argument("--seeded-cases", type=int, default=5000, help="cases per seeded sensitivity run")
    parser.add_argument("--seeded-seeds", default="1,1001,2001,3001,4001", help="seeded sensitivity seeds")
    parser.add_argument("--ablation-cases", type=int, default=2000, help="cases per module-ablation run")
    parser.add_argument("--ablation-seeds", default="1,1001,2001", help="module-ablation seeds")
    parser.add_argument("--comparison-cases", type=int, default=2000, help="cases per baseline/comparison run")
    parser.add_argument("--comparison-seeds", default="1,1001,2001", help="baseline/comparison seeds")
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
    tracks = (
        set(FINAL_PROTOCOL_TRACKS)
        if args.track == "all"
        else {args.track}
    )
    if "validation" in tracks:
        commands.append(short_validation_command(args))
    if "live" in tracks:
        commands.extend(live_discovery_commands(args))
    if "historical" in tracks:
        commands.extend(historical_replay_commands(args))
    if "seeded" in tracks:
        commands.append(seeded_sensitivity_command(args))
    if "ablation" in tracks:
        commands.append(module_ablation_command(args))
    if "comparison" in tracks:
        commands.append(method_comparison_command(args))
    return commands


def _append_experiment_meta(cmd: list[str], meta: dict[str, object]) -> None:
    cmd.extend(["--experiment-meta", json.dumps(meta, ensure_ascii=False, sort_keys=True)])


def short_validation_command(args: argparse.Namespace) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.validation_cases))),
        "--seeds",
        str(args.validation_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_VALIDATION_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_VALIDATION_MATRIX.target_suites),
        "--evidence-mode",
        "validation",
        "--run-theme",
        "final-validation-smoke",
        "--paper-notes",
        (
            "Short pre-freeze validation over live target families; inspect run-health, "
            "classify-run, experiment-summary, and methodology-report before starting 24h runs."
        ),
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        "compact",
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    _append_experiment_meta(
        cmd,
        FINAL_VALIDATION_MATRIX.command_experiment_meta(
            target_suites=FINAL_VALIDATION_MATRIX.target_suites,
        ),
    )
    return FinalCommand(
        track="validation",
        name="short_validation_smoke",
        command=cmd,
        purpose=(
            "Run a short validation-mode smoke matrix before the frozen 24h campaigns to catch "
            "adapter, oracle, preflight, classification, and evidence-pipeline noise."
        ),
        count_as_real_bugs=False,
        expected_output=(
            "short experiment manifest plus run-health/classify-run/experiment-summary/"
            "methodology-report checks; do not count candidates as final 24h bug evidence"
        ),
        notes=(
            "If this track exposes harness noise, fix it before freezing and regenerate all "
            "final plans. Validation keeps enable_replay_bug=false and only gates readiness."
        ),
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=FINAL_VALIDATION_MATRIX.command_experiment_meta(
            target_suites=FINAL_VALIDATION_MATRIX.target_suites,
        ),
    )


def live_discovery_commands(args: argparse.Namespace) -> list[FinalCommand]:
    commands = []
    replay_sources = replay_source_issues()
    for campaign in FINAL_LIVE_DISCOVERY_MATRIX.campaigns:
        suite, preset, purpose = campaign.suite, campaign.preset, campaign.purpose
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
        ]
        if args.skip_run_reports:
            cmd.append("--skip-run-reports")
        experiment_meta = FINAL_LIVE_DISCOVERY_MATRIX.command_experiment_meta_for_campaign(campaign)
        _append_experiment_meta(cmd, experiment_meta)
        commands.append(
            FinalCommand(
                track="live",
                name=f"{suite}:{preset}",
                command=cmd,
                purpose=purpose,
                count_as_real_bugs=True,
                expected_output=(
                    "runs/experiment-*.json plus experiment-summary/analyze-experiment/"
                    "methodology-report outputs"
                ),
                notes=(
                    "Fresh/latest mode keeps enable_replay_bug=false and filters known replay probes. "
                    "Run without changing generator/oracle code after inspecting findings. "
                    "Count unique candidate bug families, then separately mark maintainer-confirmed bugs."
                ),
                replay_bug_policy={
                    "enable_replay_bug": False,
                    "source_issues": replay_sources,
                },
                experiment_meta=experiment_meta,
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
        ]
        if args.skip_run_reports:
            cmd.append("--skip-run-reports")
        experiment_meta = build_historical_experiment_meta(spec)
        _append_experiment_meta(cmd, experiment_meta)
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
                experiment_meta=experiment_meta,
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
    experiment_meta = build_historical_experiment_meta(spec)
    cmd.extend(["--experiment-meta", json.dumps(experiment_meta, ensure_ascii=False, sort_keys=True)])
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
        experiment_meta=experiment_meta,
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
        ",".join(variant.preset for variant in FINAL_SEEDED_SENSITIVITY_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_SEEDED_SENSITIVITY_MATRIX.target_suites),
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
        "--skip-run-reports",
    ]
    experiment_meta = FINAL_SEEDED_SENSITIVITY_MATRIX.command_experiment_meta(
        target_suites=FINAL_SEEDED_SENSITIVITY_MATRIX.target_suites,
    )
    _append_experiment_meta(cmd, experiment_meta)
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
        experiment_meta=experiment_meta,
    )


def module_ablation_command(args: argparse.Namespace) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.ablation_cases))),
        "--seeds",
        str(args.ablation_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_MODULE_ABLATION_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_MODULE_ABLATION_MATRIX.target_suites),
        "--evidence-mode",
        "ablation",
        "--run-theme",
        "final-ablation-modules",
        "--paper-notes",
        "Module ablation for generator typing, normalizer, feedback, reducer, and oracle composition.",
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    experiment_meta = FINAL_MODULE_ABLATION_MATRIX.command_experiment_meta(
        target_suites=FINAL_MODULE_ABLATION_MATRIX.target_suites,
    )
    _append_experiment_meta(cmd, experiment_meta)
    return FinalCommand(
        track="ablation",
        name="module_ablation",
        command=cmd,
        purpose=(
            "Quantify sensitivity of type-aware generation, semantic normalization, feedback, "
            "reducer, and oracle composition across core target families."
        ),
        count_as_real_bugs=False,
        expected_output=(
            "experiment manifest plus experiment-summary/analyze-experiment/"
            "analyze-ablation-audit/methodology-report outputs"
        ),
        notes=(
            "Use for RQ ablation and baseline comparison tables. Candidate bugs from this track "
            "require the same live confirmation pipeline before they can be counted as real bugs."
        ),
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=experiment_meta,
    )


def method_comparison_command(args: argparse.Namespace) -> FinalCommand:
    cmd = [
        str(DATADIFF),
        "experiment",
        "--cases",
        str(max(1, int(args.comparison_cases))),
        "--seeds",
        str(args.comparison_seeds),
        "--presets",
        ",".join(variant.preset for variant in FINAL_COMPARISON_MATRIX.variants),
        "--target-suites",
        ",".join(FINAL_COMPARISON_MATRIX.target_suites),
        "--evidence-mode",
        "comparison",
        "--run-theme",
        "final-baseline-and-scope-comparison",
        "--paper-notes",
        (
            "Baseline and related-scope comparison: SQL/DBMS-style target suites versus "
            "cross-ecosystem DataFrame/Arrow/SQL target suites under the same harness."
        ),
        "--replay-bug-source-issues",
        ",".join(replay_source_issues()),
        "--artifact-limit",
        str(max(0, int(args.artifact_limit))),
        "--log-level",
        str(args.log_level),
        "--jobs",
        jobs_arg(args.jobs),
        "--skip-run-reports",
    ]
    experiment_meta = FINAL_COMPARISON_MATRIX.command_experiment_meta(
        target_suites=FINAL_COMPARISON_MATRIX.target_suites,
    )
    _append_experiment_meta(cmd, experiment_meta)
    return FinalCommand(
        track="comparison",
        name="baseline_and_related_scope",
        command=cmd,
        purpose=(
            "Compare random/guided/metamorphic/workflow presets and SQL/query-engine-only "
            "scope against the cross-ecosystem DataDiffFuzz scope."
        ),
        count_as_real_bugs=False,
        expected_output="experiment manifest plus baseline, methodology, and space/time efficiency analysis outputs",
        notes=(
            "This is not a reimplementation of SQLancer/SQUIRREL; it is a controlled scope "
            "baseline inside the same runner, used to isolate what DataFrame/Arrow/cross-family "
            "coverage adds beyond SQL/query-engine-oriented testing."
        ),
        replay_bug_policy={
            "enable_replay_bug": False,
            "source_issues": replay_source_issues(),
        },
        experiment_meta=experiment_meta,
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
            "validation_counts": "Short validation runs gate the harness before 24h runs and do not count as real bugs.",
            "ablation_counts": "Ablation/comparison runs support RQ tables and do not directly count as real bugs.",
            "paper_run_journal": (
                "Every final-plan command keeps paper-run-journal recording enabled so each counted or "
                "paper-facing support run is appended to reports/paper-run-journal.jsonl and .md."
            ),
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
