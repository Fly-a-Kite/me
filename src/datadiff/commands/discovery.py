from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datadiff.backend_sampling_cli import add_backend_sampling_flags
from datadiff.candidate_pool_sampling_cli import add_candidate_pool_sampling_flags


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class DiscoveryCommandHandlers:
    discovery_run: CommandHandler
    discovery_campaign: CommandHandler
    discovery_campaign_status: CommandHandler
    discovery_campaign_aggregate: CommandHandler
    candidate_pipeline: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: DiscoveryCommandHandlers,
    target_suites: Sequence[str],
    list_audit_probe_ids: Callable[[], Sequence[str]],
    default_discovery_lane_ids: Sequence[str],
    default_campaign_history_window: int,
    default_campaign_score_weights: Mapping[str, float],
    candidate_pipeline_output_dir_default: str,
) -> None:
    p_discovery_run = subparsers.add_parser(
        "discovery-run",
        help="run the integrated latest-version discovery workflow: audit, fuzz, report, classify, and manifest",
    )
    p_discovery_run.add_argument("--cases", type=int, default=500, help="fresh fuzz case budget")
    p_discovery_run.add_argument("--duration", default=None, help="optional wall-clock budget such as 10m or 24h")
    p_discovery_run.add_argument("--seed", type=int, default=1)
    p_discovery_run.add_argument(
        "--target-suite",
        choices=sorted(target_suites),
        default="latest_all_engines",
        help="backend target suite for the fresh fuzz stage",
    )
    p_discovery_run.add_argument(
        "--backends",
        default=None,
        help="explicit comma-separated backend targets; overrides --target-suite",
    )
    p_discovery_run.add_argument(
        "--preset",
        default="live_deep_organic",
        help="experiment preset for the fresh fuzz stage; defaults to live_deep_organic",
    )
    p_discovery_run.add_argument(
        "--probes",
        default="",
        help=f"comma-separated audit probe ids; defaults to all: {','.join(list_audit_probe_ids())}",
    )
    p_discovery_run.add_argument("--skip-bug-audit", action="store_true", help="skip deterministic audit stage")
    p_discovery_run.add_argument(
        "--write-issues",
        dest="write_issues",
        action="store_true",
        default=True,
        help="write audit candidate issue drafts into --issue-dir; enabled by default",
    )
    p_discovery_run.add_argument(
        "--no-write-issues",
        dest="write_issues",
        action="store_false",
        help="do not write audit issue drafts",
    )
    p_discovery_run.add_argument(
        "--issue-dir",
        default="new_issue/generated",
        help="directory for generated audit issue drafts",
    )
    p_discovery_run.add_argument(
        "--overwrite-issues",
        dest="overwrite_issues",
        action="store_true",
        default=True,
        help="overwrite existing generated audit issue drafts; enabled by default",
    )
    p_discovery_run.add_argument(
        "--no-overwrite-issues",
        dest="overwrite_issues",
        action="store_false",
        help="keep existing generated audit issue drafts",
    )
    p_discovery_run.add_argument(
        "--output-manifest",
        default="new_issue/generated/discovery-run-manifest.json",
        help="portable manifest for the integrated discovery run",
    )
    p_discovery_run.add_argument("--skip-run-report", action="store_true", help="skip markdown/csv report generation")
    p_discovery_run.add_argument("--classify-limit", type=int, default=3, help="example count per triage verdict")
    p_discovery_run.add_argument(
        "--refresh-classification",
        action="store_true",
        help="recompute differential findings from stored normalized outputs before classifying",
    )
    p_discovery_run.add_argument(
        "--extra-known-saturated-bug-families",
        default="",
        help="additional comma-separated root@backend families to exclude from fresh counts",
    )
    p_discovery_run.add_argument(
        "--candidate-recheck-count",
        type=int,
        default=None,
        help="override preset candidate recheck count",
    )
    p_discovery_run.add_argument(
        "--metamorphic-variant-limit",
        type=int,
        default=None,
        help="override preset metamorphic variant limit",
    )
    p_discovery_run.add_argument("--artifact-limit", type=int, default=None)
    p_discovery_run.add_argument("--no-compress-run-log", action="store_true")
    p_discovery_run.add_argument(
        "--enable-parallel-backend-execution",
        action="store_true",
        help="enable the non-promoted parallel backend arm",
    )
    p_discovery_run.add_argument(
        "--disable-parallel-backend-execution",
        action="store_true",
        help="explicitly retain sequential backend execution",
    )
    add_backend_sampling_flags(p_discovery_run)
    add_candidate_pool_sampling_flags(p_discovery_run)
    p_discovery_run.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="fresh fuzz JSONL detail level",
    )
    p_discovery_run.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when the fuzz stage finds a non-saturated candidate family",
    )
    p_discovery_run.add_argument(
        "--skip-candidate-pipeline",
        action="store_true",
        help="skip the automatic freeze/recheck/reduce/dedup/issue-readiness pipeline for fresh candidates",
    )
    p_discovery_run.add_argument("--candidate-pipeline-recheck-attempts", type=int, default=2)
    p_discovery_run.add_argument("--candidate-pipeline-output-dir", default=candidate_pipeline_output_dir_default)
    p_discovery_run.add_argument("--no-candidate-pipeline-reduce", action="store_true")
    p_discovery_run.add_argument("--no-candidate-pipeline-standalone-reproducer", action="store_true")
    p_discovery_run.set_defaults(func=handlers.discovery_run)

    p_discovery_campaign = subparsers.add_parser(
        "discovery-campaign",
        help="run multiple narrow latest-version discovery lanes and write one evidence manifest",
    )
    p_discovery_campaign.add_argument("--cases", type=int, default=100, help="case budget per lane/seed")
    p_discovery_campaign.add_argument("--duration", default=None, help="optional wall-clock budget per lane/seed")
    p_discovery_campaign.add_argument("--seeds", default="1", help="comma-separated seeds for every selected lane")
    p_discovery_campaign.add_argument(
        "--lanes",
        default="",
        help=f"comma-separated lane ids; defaults to: {','.join(default_discovery_lane_ids)}",
    )
    p_discovery_campaign.add_argument("--list-lanes", action="store_true", help="print available discovery lanes and exit")
    p_discovery_campaign.add_argument("--json", action="store_true", help="with --list-lanes, emit lane catalog as JSON")
    p_discovery_campaign.add_argument(
        "--probes",
        default="",
        help=f"comma-separated audit probe ids; defaults to all: {','.join(list_audit_probe_ids())}",
    )
    p_discovery_campaign.add_argument("--skip-bug-audit", action="store_true", help="skip deterministic audit stage")
    p_discovery_campaign.add_argument(
        "--write-issues",
        dest="write_issues",
        action="store_true",
        default=True,
        help="write audit candidate issue drafts into --issue-dir; enabled by default",
    )
    p_discovery_campaign.add_argument(
        "--no-write-issues",
        dest="write_issues",
        action="store_false",
        help="do not write audit issue drafts",
    )
    p_discovery_campaign.add_argument("--issue-dir", default="new_issue/generated")
    p_discovery_campaign.add_argument(
        "--overwrite-issues",
        dest="overwrite_issues",
        action="store_true",
        default=True,
        help="overwrite existing generated audit issue drafts; enabled by default",
    )
    p_discovery_campaign.add_argument(
        "--no-overwrite-issues",
        dest="overwrite_issues",
        action="store_false",
        help="keep existing generated audit issue drafts",
    )
    p_discovery_campaign.add_argument(
        "--output-manifest",
        default="new_issue/generated/discovery-campaign-manifest.json",
        help="portable manifest for the guided discovery campaign",
    )
    p_discovery_campaign.add_argument("--skip-run-report", action="store_true", help="skip markdown/csv report generation")
    p_discovery_campaign.add_argument("--classify-limit", type=int, default=3, help="example count per triage verdict")
    p_discovery_campaign.add_argument(
        "--refresh-classification",
        action="store_true",
        help="recompute differential findings from stored normalized outputs before classifying",
    )
    p_discovery_campaign.add_argument(
        "--extra-known-saturated-bug-families",
        default="",
        help="additional comma-separated root@backend families to exclude from fresh counts",
    )
    p_discovery_campaign.add_argument("--candidate-recheck-count", type=int, default=None)
    p_discovery_campaign.add_argument("--metamorphic-variant-limit", type=int, default=None)
    p_discovery_campaign.add_argument("--artifact-limit", type=int, default=None)
    p_discovery_campaign.add_argument("--no-compress-run-log", action="store_true")
    p_discovery_campaign.add_argument(
        "--enable-parallel-backend-execution",
        action="store_true",
        help="enable the non-promoted parallel backend arm",
    )
    p_discovery_campaign.add_argument(
        "--disable-parallel-backend-execution",
        action="store_true",
        help="explicitly retain sequential backend execution",
    )
    add_backend_sampling_flags(p_discovery_campaign)
    add_candidate_pool_sampling_flags(p_discovery_campaign)
    p_discovery_campaign.add_argument(
        "--log-level",
        choices=["full", "compact", "minimal"],
        default="compact",
        help="fresh fuzz JSONL detail level",
    )
    p_discovery_campaign.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when any lane finds a non-saturated candidate family",
    )
    p_discovery_campaign.add_argument(
        "--watch-health",
        action="store_true",
        help="stop remaining lanes after any completed lane/seed run contains a bug row or organic fresh candidate",
    )
    p_discovery_campaign.add_argument("--lane-history-window", type=int, default=default_campaign_history_window)
    p_discovery_campaign.add_argument(
        "--lane-yield-weight",
        type=float,
        default=default_campaign_score_weights["yield_rate"],
    )
    p_discovery_campaign.add_argument(
        "--lane-novelty-weight",
        type=float,
        default=default_campaign_score_weights["novelty_rate"],
    )
    p_discovery_campaign.add_argument(
        "--lane-false-positive-penalty",
        type=float,
        default=default_campaign_score_weights["false_positive_penalty"],
    )
    p_discovery_campaign.add_argument(
        "--skip-candidate-pipeline",
        action="store_true",
        help="skip the automatic freeze/recheck/reduce/dedup/issue-readiness pipeline for fresh candidates",
    )
    p_discovery_campaign.add_argument("--candidate-pipeline-recheck-attempts", type=int, default=2)
    p_discovery_campaign.add_argument("--candidate-pipeline-output-dir", default=candidate_pipeline_output_dir_default)
    p_discovery_campaign.add_argument("--no-candidate-pipeline-reduce", action="store_true")
    p_discovery_campaign.add_argument("--no-candidate-pipeline-standalone-reproducer", action="store_true")
    p_discovery_campaign.set_defaults(func=handlers.discovery_campaign)

    p_discovery_campaign_status = subparsers.add_parser(
        "discovery-campaign-status",
        help="summarize a running or completed discovery-campaign manifest and its latest observed run health",
    )
    p_discovery_campaign_status.add_argument("--manifest", default="new_issue/generated/discovery-campaign-manifest.json")
    p_discovery_campaign_status.add_argument(
        "--limit",
        type=int,
        default=3,
        help="candidate examples to show from latest run",
    )
    p_discovery_campaign_status.add_argument("--json", action="store_true", help="emit machine-readable status JSON")
    p_discovery_campaign_status.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help=(
            "exit with code 2 when the discovery-campaign manifest or latest observed run "
            "contains an unsaturated organic candidate"
        ),
    )
    p_discovery_campaign_status.add_argument(
        "--fail-on-bug",
        action="store_true",
        help="exit with code 2 when the latest observed run contains any status=bug row",
    )
    p_discovery_campaign_status.set_defaults(func=handlers.discovery_campaign_status)

    p_discovery_campaign_aggregate = subparsers.add_parser(
        "discovery-campaign-aggregate",
        help="aggregate multiple discovery-campaign manifests into one long-run quality view",
    )
    p_discovery_campaign_aggregate.add_argument(
        "--manifests",
        default="new_issue/generated/discovery-campaign*.json",
        help="comma-separated manifest files or glob patterns",
    )
    p_discovery_campaign_aggregate.add_argument(
        "--limit",
        type=int,
        default=10,
        help="maximum fresh families and lane rows to include",
    )
    p_discovery_campaign_aggregate.add_argument("--output", default="", help="optional JSON output path")
    p_discovery_campaign_aggregate.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    p_discovery_campaign_aggregate.set_defaults(func=handlers.discovery_campaign_aggregate)

    p_candidate_pipeline = subparsers.add_parser(
        "candidate-pipeline",
        help="freeze fresh candidates and run recheck/reduce/dedup/issue-readiness automatically",
    )
    p_candidate_pipeline.add_argument("--manifest", default=None)
    p_candidate_pipeline.add_argument(
        "--evidence-files",
        default="",
        help="comma-separated fresh candidate evidence JSON files; overrides --manifest discovery when provided",
    )
    p_candidate_pipeline.add_argument("--output-dir", default=candidate_pipeline_output_dir_default)
    p_candidate_pipeline.add_argument("--recheck-attempts", type=int, default=2)
    p_candidate_pipeline.add_argument("--no-reduce", action="store_true")
    p_candidate_pipeline.add_argument("--no-standalone-reproducer", action="store_true")
    p_candidate_pipeline.add_argument("--json", action="store_true")
    p_candidate_pipeline.add_argument(
        "--fail-on-ready",
        action="store_true",
        help="exit with code 2 when the pipeline produces any ready-to-submit draft",
    )
    p_candidate_pipeline.set_defaults(func=handlers.candidate_pipeline)
