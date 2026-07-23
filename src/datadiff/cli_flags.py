from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Callable

from datadiff.backend_sampling_cli import add_backend_sampling_flags
from datadiff.candidate_pool_sampling_cli import add_candidate_pool_sampling_flags
from datadiff.method_arms import (
    DEFAULT_METHOD_ARM_ID,
    live_method_arm_ids,
    research_control_arm_ids,
)


def add_ablation_flags(
    parser: argparse.ArgumentParser,
    *,
    parse_adaptive_components_func: Callable[[str], set[str]],
    adaptive_components: Sequence[str],
) -> None:
    parser.add_argument(
        "--method-arm",
        choices=live_method_arm_ids(),
        default=DEFAULT_METHOD_ARM_ID,
        help=(
            "registered research arm; use this instead of assembling implicit method "
            "booleans for paper ablations"
        ),
    )
    parser.add_argument(
        "--research-control-arm",
        choices=research_control_arm_ids(),
        default="",
        help=(
            "explicit frozen/ablation control; research controls are separate from "
            "the live P8 method registry"
        ),
    )
    parser.add_argument("--disable-type-aware-generation", action="store_true")
    parser.add_argument("--disable-normalizer", action="store_true")
    parser.add_argument("--disable-differential-oracle", action="store_true")
    parser.add_argument("--enable-metamorphic-oracle", action="store_true")
    parser.add_argument(
        "--enable-witness-oracle",
        action="store_true",
        help=(
            "enable experimental SQLancer/PQS-inspired witness-contract checking; "
            "off by default and should be reported as a separate experiment track"
        ),
    )
    parser.add_argument("--disable-feedback", action="store_true")
    parser.add_argument(
        "--enable-replay-bug",
        action="store_true",
        help="allow known issue-replay cases; default fresh mode filters submitted or replay-only bug targets",
    )
    parser.add_argument("--enable-reducer", action="store_true")
    parser.add_argument("--disable-artifact", action="store_true")
    parser.add_argument(
        "--enable-parallel-backend-execution",
        action="store_true",
        help=(
            "enable the non-promoted parallel backend arm; the P4.8-B production "
            "default is sequential"
        ),
    )
    parser.add_argument(
        "--disable-parallel-backend-execution",
        action="store_true",
        help="explicitly retain the sequential P4.8-B production default",
    )
    parser.add_argument(
        "--disable-backend-session-reuse",
        action="store_true",
        help=(
            "construct fresh backend instances for each execution call; use as the "
            "P4.2 isolation/control arm"
        ),
    )
    add_backend_sampling_flags(parser)
    add_candidate_pool_sampling_flags(parser)
    parser.add_argument("--disable-preflight-validation", action="store_true")
    parser.add_argument("--disable-preflight-repair", action="store_true")
    parser.add_argument("--persist-feedback-corpus", action="store_true")
    parser.add_argument(
        "--feedback-persist-limit",
        type=int,
        default=4096,
        help="maximum interesting feedback cases to write to corpus/interesting for this run",
    )
    parser.add_argument(
        "--enable-local-source-scheduler",
        action="store_true",
        help="adaptively choose between generated candidates and feedback mutations within a run",
    )
    parser.add_argument(
        "--local-source-exploration-weight",
        type=float,
        default=0.5,
        help="exploration weight for the within-run generated-vs-feedback source scheduler",
    )
    parser.add_argument(
        "--disable-adaptive-components",
        type=parse_adaptive_components_func,
        default="",
        help=(
            "comma-separated adaptive components to disable for ablation: "
            + ",".join(adaptive_components)
        ),
    )
    parser.add_argument("--no-compress-run-log", action="store_true")
    parser.add_argument(
        "--artifact-limit",
        type=int,
        default=None,
        help="maximum bug artifact directories to write for this run; 0 keeps only run-log finding summaries",
    )
    parser.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=4,
        help="maximum metamorphic variants to execute per base case",
    )
    parser.add_argument(
        "--metamorphic-relation-order",
        default="",
        help="comma-separated MR relation priority order used before adaptive per-case MR learning",
    )
    parser.add_argument(
        "--semantic-objective-learning-weight",
        type=float,
        default=0.0,
        help="per-case semantic objective contextual-learning weight; 0 keeps objective selection observational",
    )
    parser.add_argument(
        "--metamorphic-relation-learning-weight",
        type=float,
        default=0.0,
        help="per-case MR type contextual-learning weight; 0 keeps configured MR order",
    )
    parser.add_argument(
        "--version-pair-learning-weight",
        type=float,
        default=0.0,
        help="per-case target-version-pair contextual-learning weight; 0 records no version-pair arm feedback",
    )
    parser.add_argument(
        "--backend-pair-learning-weight",
        type=float,
        default=0.0,
        help="per-case backend-pair contextual-learning weight; 0 keeps pair priority observational",
    )
    parser.add_argument(
        "--backend-pair-priority-limit",
        type=int,
        default=3,
        help="maximum backend pairs to expose as adaptive priority for recheck/metamorphic scheduling",
    )
    parser.add_argument(
        "--candidate-recheck-count",
        type=int,
        default=0,
        help="rerun finding cases this many times and mark non-reproduced findings as false positives",
    )
    parser.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="run JSONL detail level; compact keeps full details only for finding rows",
    )
    parser.add_argument(
        "--strategy-snapshot",
        default="",
        help="path to a frozen dynamic strategy snapshot used by classification/reproduction logic",
    )
    parser.add_argument(
        "--strategy-learning",
        default="",
        help="path to a strategy-learning ledger used for evidence-driven updates outside frozen runs",
    )
    parser.add_argument(
        "--exploration-objective-rules",
        default="",
        help=(
            "JSON or @path defining neutral exploration objective rules; "
            "each rule has objective, exact_features, prefix_features, and fragments"
        ),
    )
    parser.add_argument(
        "--freeze-strategy-snapshot",
        action="store_true",
        help="treat the configured strategy snapshot as frozen and disable runtime learning drift for final runs",
    )


def add_guidance_flags(
    parser: argparse.ArgumentParser,
    *,
    default_strategy: str,
    default_candidate_pool: int,
) -> None:
    parser.add_argument("--strategy", choices=["random", "guided"], default=default_strategy)
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=default_candidate_pool,
        help="number of cheap generated candidates scored before executing one case",
    )
    parser.add_argument(
        "--targets",
        default="",
        help=(
            "comma-separated guided targets such as groupby,filter,mutate,sort_limit,"
            "nulls,strings,numeric,edge_float,aggregation,join,running_sum,sortedness,expressions,casts"
        ),
    )
    parser.add_argument(
        "--disable-family-saturation",
        action="store_true",
        help="disable repeated candidate bug family downweighting in guided selection and online rewards",
    )
    parser.add_argument(
        "--family-saturation-threshold",
        type=int,
        default=8,
        help="candidate bug family hit count where guidance starts treating the family as saturated",
    )
    parser.add_argument(
        "--family-saturation-penalty",
        type=float,
        default=1.25,
        help="score penalty scale for predicted cases in saturated candidate bug families",
    )
    parser.add_argument(
        "--saturated-family-reward",
        type=float,
        default=0.02,
        help="online reward assigned to a candidate bug family after saturation",
    )
    parser.add_argument(
        "--known-saturated-bug-families",
        default="",
        help="comma-separated root@backend families already considered saturated before this run",
    )
    parser.add_argument(
        "--replay-bug-source-issues",
        default="",
        help="comma-separated upstream issue URLs treated as known replay bugs in fresh mode",
    )
    parser.add_argument(
        "--issue-replay-saturation-threshold",
        type=int,
        default=1,
        help="issue-replay family hit count where guidance starts downweighting repeated replay probes",
    )
    parser.add_argument(
        "--issue-replay-saturation-penalty",
        type=float,
        default=1.0,
        help="score penalty scale for predicted cases in saturated issue-replay families",
    )
    parser.add_argument(
        "--issue-replay-global-saturation-threshold",
        type=int,
        default=4,
        help="total issue-replay candidate bug count where guidance starts downweighting replay probes",
    )
    parser.add_argument(
        "--issue-replay-global-saturation-penalty",
        type=float,
        default=1.5,
        help="score penalty scale for replay probes after the global replay budget is saturated",
    )
    parser.add_argument(
        "--issue-inspired-source-saturation-threshold",
        type=int,
        default=3,
        help="candidate bug count per source issue where guidance starts downweighting issue-inspired cases",
    )
    parser.add_argument(
        "--issue-inspired-source-saturation-penalty",
        type=float,
        default=1.25,
        help="score penalty scale for issue-inspired cases after their source issue is saturated",
    )


def add_target_suite_flags(
    parser: argparse.ArgumentParser,
    *,
    target_suites: dict[str, object],
) -> None:
    parser.add_argument(
        "--target-suite",
        choices=sorted(target_suites),
        default="core",
        help="backend target suite to execute when --backends is not provided",
    )
    parser.add_argument(
        "--backends",
        default=None,
        help="explicit comma-separated backend targets; overrides --target-suite",
    )


def add_paper_journal_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-theme",
        default="",
        help="paper-facing run theme recorded in reports/paper-run-journal.*",
    )
    parser.add_argument(
        "--paper-notes",
        default="",
        help="short paper-facing notes recorded with the run journal entry",
    )
    parser.add_argument(
        "--skip-paper-journal",
        action="store_true",
        help="skip paper-run journal writes for IO-sensitive long runs; summarize later from the manifest/run log",
    )
