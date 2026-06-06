from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class ExperimentCommandHandlers:
    experiment: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: ExperimentCommandHandlers,
    experiment_evidence_modes: Sequence[str],
    adaptive_components: Sequence[str],
    parse_jobs: Callable[[str], int | str],
    parse_adaptive_components: Callable[[str | list[str] | tuple[str, ...] | set[str] | None], set[str]],
    add_target_suite_flags: Callable[[argparse.ArgumentParser], None],
    add_paper_journal_flags: Callable[[argparse.ArgumentParser], None],
) -> None:
    p_exp = subparsers.add_parser("experiment", help="run repeatable ablation experiment matrix")
    p_exp.add_argument("--cases", type=int, default=None, help="maximum cases; defaults to 100 when --duration is absent")
    p_exp.add_argument(
        "--duration",
        default=None,
        help=(
            "optional wall-clock budget such as 10s, 5m, 24h; adaptive schedule "
            "treats this as the total matrix budget, while static schedules treat it per run"
        ),
    )
    p_exp.add_argument("--seeds", default="1,1001,2001")
    p_exp.add_argument(
        "--evidence-mode",
        choices=["auto", *experiment_evidence_modes],
        default="auto",
        help=(
            "experiment evidence layer: validation smoke, live latest-version finding, "
            "historical fixed-bug replay, seeded fault sensitivity, module ablation, "
            "or baseline/related-scope comparison"
        ),
    )
    p_exp.add_argument(
        "--known-bug-id",
        default="",
        help="identifier for a replayed historical/upstream bug when --evidence-mode=historical",
    )
    p_exp.add_argument(
        "--profile-pool",
        default="",
        help="comma-separated generator profiles for adaptive per-case profile selection inside each experiment run",
    )
    p_exp.add_argument(
        "--profile-learning-weight",
        type=float,
        default=0.0,
        help="per-case generator profile contextual-learning weight; 0 keeps each preset's fixed generator profile",
    )
    p_exp.add_argument(
        "--target-version",
        default="",
        help="target dependency version or commit used by a historical replay run",
    )
    p_exp.add_argument(
        "--version-pair-pool",
        default="",
        help="comma-separated target-version pairs for adaptive per-case cross-version selection",
    )
    p_exp.add_argument(
        "--fixed-version",
        default="",
        help="fixed dependency version or commit paired with --target-version for cross-version learning",
    )
    p_exp.add_argument(
        "--semantic-objective-learning-weight",
        type=float,
        default=0.0,
        help="per-case semantic objective contextual-learning weight; 0 keeps objective selection observational",
    )
    p_exp.add_argument(
        "--enable-metamorphic-oracle",
        action="store_true",
        help="enable metamorphic oracle execution across experiment runs without requiring a metamorphic preset",
    )
    p_exp.add_argument(
        "--metamorphic-relation-learning-weight",
        type=float,
        default=0.0,
        help="per-case MR type contextual-learning weight; 0 keeps configured MR order",
    )
    p_exp.add_argument(
        "--metamorphic-relation-order",
        default="",
        help="comma-separated MR relation priority order used before adaptive per-case MR learning",
    )
    p_exp.add_argument(
        "--version-pair-learning-weight",
        type=float,
        default=0.0,
        help="per-case target-version-pair contextual-learning weight; 0 records no version-pair arm feedback",
    )
    p_exp.add_argument(
        "--backend-pair-learning-weight",
        type=float,
        default=0.0,
        help="per-case backend-pair contextual-learning weight; 0 keeps pair priority observational",
    )
    p_exp.add_argument(
        "--backend-pair-priority-limit",
        type=int,
        default=3,
        help="maximum backend pairs to expose as adaptive priority for recheck/metamorphic scheduling",
    )
    add_target_suite_flags(p_exp)
    p_exp.add_argument(
        "--target-suites",
        default=None,
        help="comma-separated backend target suites to run as an extra experiment dimension",
    )
    p_exp.add_argument(
        "--artifact-limit",
        type=int,
        default=None,
        help="maximum bug artifact directories to write for each preset run",
    )
    p_exp.add_argument("--log-level", choices=["full", "compact", "minimal"], default="compact")
    p_exp.add_argument(
        "--jobs",
        type=parse_jobs,
        default="auto",
        help="number of experiment matrix runs to execute in parallel, or 'auto'",
    )
    p_exp.add_argument(
        "--max-parallel-cost",
        type=float,
        default=None,
        help="cost-token budget for concurrently running experiment jobs; defaults to a CPU-based budget",
    )
    p_exp.add_argument(
        "--schedule",
        choices=["matrix_order", "longest_first", "adaptive"],
        default=None,
        help="experiment scheduler; adaptive shares the matrix budget across runs",
    )
    p_exp.add_argument(
        "--batch-cases",
        type=int,
        default=None,
        help="adaptive-schedule batch size in cases; ignored by static schedules",
    )
    p_exp.add_argument(
        "--batch-duration",
        default=None,
        help="adaptive-schedule batch wall-clock budget such as 30s; ignored by static schedules",
    )
    p_exp.add_argument(
        "--warmup-batches",
        type=int,
        default=1,
        help="minimum adaptive batches to allocate to each arm before exploitation",
    )
    p_exp.add_argument(
        "--exploration-weight",
        type=float,
        default=0.75,
        help="adaptive scheduler exploration weight",
    )
    p_exp.add_argument(
        "--group-fairness-weight",
        type=float,
        default=0.40,
        help="adaptive scheduler bonus for underrepresented target-suite/preset groups",
    )
    p_exp.add_argument(
        "--max-group-pull-gap",
        type=int,
        default=3,
        help="adaptive scheduler rebalances once a target-suite/preset group trails by this many pulls",
    )
    p_exp.add_argument(
        "--adaptive-learning-weight",
        type=float,
        default=0.0,
        help="adaptive scheduler contextual-learning score weight; 0 keeps legacy adaptive scheduling",
    )
    p_exp.add_argument(
        "--scheduler-annealing-temperature",
        type=float,
        default=0.0,
        help="initial adaptive scheduler annealing temperature; 0 keeps deterministic greedy selection",
    )
    p_exp.add_argument(
        "--scheduler-annealing-decay",
        type=float,
        default=0.985,
        help="per-completed-batch decay for adaptive scheduler annealing temperature",
    )
    p_exp.add_argument(
        "--scheduler-annealing-min-temperature",
        type=float,
        default=0.02,
        help="minimum nonzero adaptive scheduler annealing temperature",
    )
    p_exp.add_argument(
        "--continual-learning-ledgers",
        default="",
        help="comma-separated version-ledger JSON files used to cold-start adaptive continual-learning priority",
    )
    p_exp.add_argument(
        "--disable-adaptive-components",
        type=parse_adaptive_components,
        default="",
        help=(
            "comma-separated adaptive components to disable for ablation: "
            + ",".join(adaptive_components)
        ),
    )
    p_exp.add_argument(
        "--enable-local-source-scheduler",
        action="store_true",
        help="enable within-run generated-vs-feedback source scheduling for non-adaptive experiment jobs",
    )
    p_exp.add_argument(
        "--local-source-exploration-weight",
        type=float,
        default=0.5,
        help="exploration weight for the within-run generated-vs-feedback source scheduler",
    )
    p_exp.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=None,
        help="maximum metamorphic variants to execute per base case",
    )
    p_exp.add_argument(
        "--enable-replay-bug",
        action="store_true",
        help="allow submitted or historical issue replay cases in experiment presets",
    )
    p_exp.add_argument(
        "--replay-bug-source-issues",
        default="",
        help="comma-separated upstream issue URLs treated as known replay bugs in fresh experiment mode",
    )
    p_exp.add_argument("--no-compress-run-log", action="store_true")
    p_exp.add_argument(
        "--disable-parallel-backend-execution",
        action="store_true",
        help="run each experiment case on configured backends sequentially for runtime ablation",
    )
    p_exp.add_argument(
        "--persist-closed-loop-state",
        action="store_true",
        help="write a resumable closed-loop learning state file for each experiment run",
    )
    p_exp.add_argument(
        "--skip-run-reports",
        action="store_true",
        help="do not write per-run markdown/csv reports during the matrix; use experiment-summary after completion",
    )
    p_exp.add_argument(
        "--presets",
        default="baseline,no_type_aware,no_normalizer,no_feedback,metamorphic,reducer",
        help="comma-separated presets",
    )
    p_exp.add_argument(
        "--experiment-meta",
        default="",
        help="JSON object describing structured experiment catalog metadata for the whole matrix",
    )
    p_exp.add_argument(
        "--strategy-snapshot",
        default="",
        help="path to a frozen dynamic strategy snapshot used by classification/reproduction logic",
    )
    p_exp.add_argument(
        "--strategy-learning",
        default="",
        help="path to a strategy-learning ledger used for evidence-driven updates outside frozen runs",
    )
    p_exp.add_argument(
        "--freeze-strategy-snapshot",
        action="store_true",
        help="treat the configured strategy snapshot as frozen and disable runtime learning drift for final runs",
    )
    add_paper_journal_flags(p_exp)
    p_exp.set_defaults(func=handlers.experiment)
