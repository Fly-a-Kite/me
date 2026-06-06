from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class ArtifactCommandHandlers:
    reproduce: CommandHandler
    validate_artifact: CommandHandler
    triage_artifact: CommandHandler
    reduce: CommandHandler
    historical_status: CommandHandler
    replay_fixture: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: ArtifactCommandHandlers,
    target_suites: Sequence[str],
    fixture_replay_evidence_modes: Sequence[str],
    add_ablation_flags: Callable[[argparse.ArgumentParser], None],
) -> None:
    p_repro = subparsers.add_parser("reproduce", help="show reproduce command for a bug artifact")
    p_repro.add_argument("--bug", required=True)
    p_repro.add_argument("--backends", default=None)
    p_repro.add_argument("--print-command", action="store_true")
    p_repro.set_defaults(func=handlers.reproduce)

    p_validate = subparsers.add_parser("validate-artifact", help="rerun a bug artifact and check finding preservation")
    p_validate.add_argument("--bug", required=True)
    p_validate.add_argument("--backends", default=None)
    p_validate.set_defaults(func=handlers.validate_artifact)

    p_triage = subparsers.add_parser("triage-artifact", help="classify a reproduced artifact for paper use")
    p_triage.add_argument("--bug", required=True)
    p_triage.add_argument("--backends", default=None)
    p_triage.add_argument("--reduce", action="store_true")
    p_triage.add_argument(
        "--reduce-ignore-roots",
        action="store_true",
        help="when reducing, preserve finding kind only instead of the original root-cause label",
    )
    p_triage.add_argument(
        "--standalone-reproducer",
        action="store_true",
        help="also write an optional standalone diagnostic script when this root cause is supported",
    )
    p_triage.set_defaults(func=handlers.triage_artifact)

    p_reduce = subparsers.add_parser("reduce", help="minimize a bug artifact while preserving findings")
    p_reduce.add_argument("--bug", required=True)
    p_reduce.add_argument("--backends", default="pandas,polars,duckdb,sqlite")
    p_reduce.add_argument(
        "--ignore-roots",
        action="store_true",
        help="preserve finding kind only instead of the original root-cause label",
    )
    p_reduce.set_defaults(func=handlers.reduce)

    p_hist = subparsers.add_parser("historical-status", help="show historical replay registry admission status")
    p_hist.add_argument(
        "--include-pending",
        action="store_true",
        help="include candidate and pending historical case studies",
    )
    p_hist.add_argument("--json", action="store_true", help="emit historical registry status as JSON")
    p_hist.set_defaults(func=handlers.historical_status)

    p_fixture = subparsers.add_parser(
        "replay-fixture",
        help="run a declared fixture-backed case through normal differential oracle and journal recording",
    )
    p_fixture.add_argument("--spec", required=True, help="fixture replay spec JSON")
    p_fixture.add_argument("--fixture", default=None, help="path to the external fixture data file")
    p_fixture.add_argument(
        "--fixture-env",
        default=None,
        help="environment variable containing the external fixture data file path",
    )
    p_fixture.add_argument("--backends", default=None, help="explicit comma-separated backend targets")
    p_fixture.add_argument(
        "--target-suite",
        choices=sorted(target_suites),
        default=None,
        help="backend target suite used when --backends is not provided; defaults to the spec target_suite",
    )
    p_fixture.add_argument(
        "--evidence-mode",
        choices=list(fixture_replay_evidence_modes),
        default="historical",
        help="paper evidence layer for this replay",
    )
    p_fixture.add_argument("--known-bug-id", default="", help="historical bug id recorded in the run journal")
    p_fixture.add_argument("--target-version", default="", help="target dependency version or commit under replay")
    p_fixture.add_argument("--run-theme", default="", help="short paper-facing run theme")
    p_fixture.add_argument("--paper-notes", default="", help="brief paper-facing run notes")
    p_fixture.add_argument(
        "--experiment-meta",
        default="",
        help="JSON object describing structured experiment metadata for this fixture replay",
    )
    add_ablation_flags(p_fixture)
    p_fixture.set_defaults(func=handlers.replay_fixture)
