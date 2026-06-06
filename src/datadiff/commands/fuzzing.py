from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class FuzzingCommandHandlers:
    fuzz: CommandHandler
    longrun: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: FuzzingCommandHandlers,
    profile_choices: Sequence[str],
    add_target_suite_flags: Callable[[argparse.ArgumentParser], None],
    add_guidance_flags: Callable[..., None],
    add_ablation_flags: Callable[[argparse.ArgumentParser], None],
    add_paper_journal_flags: Callable[[argparse.ArgumentParser], None],
) -> None:
    p_fuzz = subparsers.add_parser("fuzz", help="run differential fuzzing")
    p_fuzz.add_argument("--cases", type=int, default=None, help="maximum cases; defaults to 100 when --duration is absent")
    p_fuzz.add_argument("--duration", default=None, help="wall-clock budget such as 10s, 5m, 24h")
    p_fuzz.add_argument("--seed", type=int, default=1)
    add_target_suite_flags(p_fuzz)
    _add_adaptive_profile_flags(p_fuzz, profile_choices=profile_choices)
    add_guidance_flags(p_fuzz, default_strategy="random", default_candidate_pool=8)
    add_ablation_flags(p_fuzz)
    add_paper_journal_flags(p_fuzz)
    p_fuzz.set_defaults(func=handlers.fuzz)

    p_long = subparsers.add_parser("longrun", help="run long-duration fuzzing and persist generated test cases")
    p_long.add_argument("--cases", type=int, default=None, help="optional maximum cases; duration is the primary budget")
    p_long.add_argument("--duration", default="24h", help="wall-clock budget such as 10m, 24h, 2d")
    p_long.add_argument("--seed", type=int, default=1)
    add_target_suite_flags(p_long)
    _add_adaptive_profile_flags(p_long, profile_choices=profile_choices)
    add_guidance_flags(p_long, default_strategy="guided", default_candidate_pool=8)
    p_long.add_argument("--case-log", default=None, help="optional JSONL path for generated test cases")
    p_long.add_argument("--checkpoint-interval", default="60s", help="checkpoint write interval")
    p_long.add_argument("--progress-interval", default="60s", help="stdout progress interval")
    p_long.add_argument("--save-cases", action="store_true", help="persist every generated test case separately")
    p_long.add_argument("--no-save-cases", action="store_true", help="do not persist generated test cases separately")
    p_long.add_argument("--quiet", action="store_true", help="suppress periodic progress output")
    add_ablation_flags(p_long)
    add_paper_journal_flags(p_long)
    p_long.set_defaults(func=handlers.longrun)


def _add_adaptive_profile_flags(
    parser: argparse.ArgumentParser,
    *,
    profile_choices: Sequence[str],
) -> None:
    parser.add_argument("--profile", choices=profile_choices, default="common")
    parser.add_argument(
        "--profile-pool",
        default="",
        help="comma-separated generator profiles for adaptive per-case profile selection",
    )
    parser.add_argument(
        "--profile-learning-weight",
        type=float,
        default=0.0,
        help="per-case generator profile contextual-learning weight; 0 keeps fixed --profile",
    )
    parser.add_argument(
        "--version-pair-pool",
        default="",
        help="comma-separated target-version pairs for adaptive per-case cross-version selection",
    )
    parser.add_argument("--target-version", default="", help="target backend/dependency version label for learning context")
    parser.add_argument("--fixed-version", default="", help="fixed/backend comparison version label for learning context")
