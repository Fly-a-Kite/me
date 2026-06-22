from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class AnalysisCommandHandlers:
    experiment_summary: CommandHandler
    analyze_experiment: CommandHandler
    analyze_seeded_sensitivity: CommandHandler
    analyze_ablation_audit: CommandHandler
    adaptive_benchmark: CommandHandler
    analyze_pattern_variants: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: AnalysisCommandHandlers,
) -> None:
    p_exp_summary = subparsers.add_parser("experiment-summary", help="summarize an experiment manifest")
    p_exp_summary.add_argument("--manifest", default=None)
    p_exp_summary.add_argument(
        "--refresh",
        action="store_true",
        help="recompute differential findings from stored normalized outputs or bug artifacts with the current oracle",
    )
    p_exp_summary.set_defaults(func=handlers.experiment_summary)

    p_exp_analysis = subparsers.add_parser(
        "analyze-experiment",
        help="compare experiment aggregate metrics against a reference preset",
    )
    p_exp_analysis.add_argument("--manifest", default=None)
    p_exp_analysis.add_argument("--reference-preset", default="baseline")
    p_exp_analysis.add_argument(
        "--compare-presets",
        default=None,
        help="optional comma-separated preset subset to compare against the reference run",
    )
    p_exp_analysis.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before analysis",
    )
    p_exp_analysis.set_defaults(func=handlers.analyze_experiment)

    p_seeded_analysis = subparsers.add_parser(
        "analyze-seeded-sensitivity",
        help="analyze expected-root detection for seeded fault experiments",
    )
    p_seeded_analysis.add_argument("--manifest", default=None)
    p_seeded_analysis.set_defaults(func=handlers.analyze_seeded_sensitivity)

    p_ablation_audit = subparsers.add_parser(
        "analyze-ablation-audit",
        help="audit candidate families and false positives introduced by ablation presets",
    )
    p_ablation_audit.add_argument("--manifest", default=None)
    p_ablation_audit.add_argument(
        "--reference-presets",
        default=None,
        help="comma-separated presets treated as the reference soundness boundary",
    )
    p_ablation_audit.add_argument(
        "--ablation-presets",
        default=None,
        help="comma-separated weakened presets whose candidates should not be counted without triage",
    )
    p_ablation_audit.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before auditing",
    )
    p_ablation_audit.set_defaults(func=handlers.analyze_ablation_audit)

    p_adaptive_benchmark = subparsers.add_parser(
        "adaptive-benchmark",
        help="quantify adaptive-learning effectiveness separately from hot-path systems overhead",
    )
    p_adaptive_benchmark.add_argument(
        "--mode",
        choices=("all", "learning", "systems", "replay"),
        default="all",
        help="run learning-effectiveness scenarios, systems microbenchmarks, real-run replay, or a combination",
    )
    p_adaptive_benchmark.add_argument(
        "--learning-rounds",
        type=int,
        default=120,
        help="episodes/rounds per learning-effectiveness scenario",
    )
    p_adaptive_benchmark.add_argument(
        "--systems-iterations",
        type=int,
        default=2000,
        help="iterations per systems microbenchmark",
    )
    p_adaptive_benchmark.add_argument(
        "--profile-iterations",
        type=int,
        default=1024,
        help="iterations for the cProfile workload",
    )
    p_adaptive_benchmark.add_argument(
        "--action-pool-size",
        type=int,
        default=24,
        help="action count used in systems microbenchmarks",
    )
    p_adaptive_benchmark.add_argument(
        "--profile-top-n",
        type=int,
        default=20,
        help="number of top cumulative cProfile rows to retain in JSON/markdown output",
    )
    p_adaptive_benchmark.add_argument(
        "--profile-output",
        default="",
        help="optional .prof output path; defaults to reports/adaptive-benchmark-*.prof when --write-report is set",
    )
    p_adaptive_benchmark.add_argument(
        "--replay-run-file",
        action="append",
        default=[],
        help="run log path to replay; may be repeated or comma-separated",
    )
    p_adaptive_benchmark.add_argument(
        "--replay-manifest",
        action="append",
        default=[],
        help="experiment manifest path to replay; may be repeated or comma-separated",
    )
    p_adaptive_benchmark.add_argument("--json", action="store_true", help="emit the benchmark payload as JSON")
    p_adaptive_benchmark.add_argument(
        "--write-report",
        action="store_true",
        help="write reports/adaptive-benchmark-*.json and .md",
    )
    p_adaptive_benchmark.add_argument(
        "--output-dir",
        default="reports",
        help="directory for --write-report output",
    )
    p_adaptive_benchmark.set_defaults(func=handlers.adaptive_benchmark)

    p_pattern_variants = subparsers.add_parser(
        "analyze-pattern-variants",
        help="analyze generated pattern variants and candidate findings in an experiment",
    )
    p_pattern_variants.add_argument("--manifest", default=None)
    p_pattern_variants.add_argument(
        "--pattern",
        choices=["null_agg_topk"],
        default="null_agg_topk",
    )
    p_pattern_variants.set_defaults(func=handlers.analyze_pattern_variants)
