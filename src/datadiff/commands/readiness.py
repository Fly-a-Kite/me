from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class ReadinessCommandHandlers:
    final_readiness: CommandHandler
    review_readiness: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: ReadinessCommandHandlers,
    default_final_readiness_manifest_limit: int,
    required_live_suites: Sequence[str],
    required_live_families: Sequence[str],
    final_readiness_thresholds: Any,
) -> None:
    p_final_ready = subparsers.add_parser(
        "final-readiness",
        help="audit final experiment breadth, depth, replay policy, and bug evidence readiness",
    )
    p_final_ready.add_argument(
        "--manifest",
        action="append",
        default=[],
        help=(
            "experiment manifest to include; may be repeated; when omitted, defaults to the latest "
            f"{default_final_readiness_manifest_limit} runs/experiment-*.json files"
        ),
    )
    p_final_ready.add_argument(
        "--extra-manifest",
        action="append",
        default=[],
        help=(
            "additional support manifest to audit alongside the default/latest run manifests; "
            "use this for reports/experiment-final-version-ledger.json"
        ),
    )
    p_final_ready.add_argument(
        "--manifest-index",
        action="append",
        default=[],
        help=(
            "JSON manifest index produced by scripts/run_final_experiments.py --execute; "
            "may be repeated and is used instead of sweeping stale runs/experiment-*.json files"
        ),
    )
    p_final_ready.add_argument(
        "--latest-manifests",
        type=int,
        default=default_final_readiness_manifest_limit,
        help="number of most-recent experiment manifests to audit when --manifest is omitted",
    )
    p_final_ready.add_argument(
        "--all-manifests",
        action="store_true",
        help="audit every runs/experiment-*.json manifest when --manifest is omitted",
    )
    p_final_ready.add_argument(
        "--summary-only",
        action="store_true",
        help="skip run-log scans and audit only manifest/meta/confirmation metadata",
    )
    p_final_ready.add_argument(
        "--full-run-log-scan",
        action="store_true",
        help="scan run logs even when --manifest is omitted; required for a final paper readiness claim",
    )
    p_final_ready.add_argument(
        "--latest-confirmation-file",
        action="append",
        default=[],
        help=(
            "JSON file with upstream-confirmed latest bug families; may be repeated; "
            "defaults to experiments/latest_confirmations.json when present"
        ),
    )
    p_final_ready.add_argument("--min-live-cases-per-suite", type=int, default=1)
    p_final_ready.add_argument("--min-live-duration-hours", type=float, default=24.0)
    p_final_ready.add_argument("--min-live-candidate-families", type=int, default=1)
    p_final_ready.add_argument("--min-confirmed-live-families", type=int, default=1)
    p_final_ready.add_argument("--min-historical-confirmed", type=int, default=2)
    p_final_ready.add_argument(
        "--required-live-suites",
        default=",".join(required_live_suites),
        help="comma-separated live target suites required by the top-level experiment policy",
    )
    p_final_ready.add_argument(
        "--required-live-families",
        default=",".join(required_live_families),
        help="comma-separated backend families required by the top-level experiment policy",
    )
    p_final_ready.add_argument("--no-require-validation", action="store_true")
    p_final_ready.add_argument("--no-require-seeded", action="store_true")
    p_final_ready.add_argument("--no-require-ablation", action="store_true")
    p_final_ready.add_argument("--no-require-comparison", action="store_true")
    p_final_ready.add_argument("--no-require-adaptive-component-ablation", action="store_true")
    p_final_ready.add_argument(
        "--min-adaptive-component-ablations",
        type=int,
        default=final_readiness_thresholds.min_adaptive_component_ablations,
    )
    p_final_ready.add_argument(
        "--required-adaptive-component-ablations",
        default=",".join(final_readiness_thresholds.required_adaptive_component_ablations),
        help=(
            "comma-separated adaptive components that must each have an ablation contrast; "
            "hyphenated aliases are accepted"
        ),
    )
    p_final_ready.add_argument("--no-require-transferability-scope", action="store_true")
    p_final_ready.add_argument(
        "--min-transfer-target-families",
        type=int,
        default=final_readiness_thresholds.min_transfer_target_families,
    )
    p_final_ready.add_argument("--no-require-cross-version-ledger", action="store_true")
    p_final_ready.add_argument(
        "--min-cross-version-ledger-versions",
        type=int,
        default=final_readiness_thresholds.min_cross_version_ledger_versions,
    )
    p_final_ready.add_argument(
        "--min-cross-version-ledger-families",
        type=int,
        default=final_readiness_thresholds.min_cross_version_ledger_families,
    )
    p_final_ready.add_argument("--no-require-cross-version-health-feedback", action="store_true")
    p_final_ready.add_argument("--no-require-runtime-efficiency", action="store_true")
    p_final_ready.add_argument(
        "--min-throughput-cases-s",
        type=float,
        default=final_readiness_thresholds.min_throughput_cases_s,
    )
    p_final_ready.add_argument(
        "--max-scheduler-feedback-share",
        type=float,
        default=final_readiness_thresholds.max_scheduler_feedback_share,
    )
    p_final_ready.add_argument(
        "--min-scheduler-feedback-cases",
        type=int,
        default=final_readiness_thresholds.min_scheduler_feedback_cases,
        help="minimum executed cases before scheduler feedback share is enforced for a run",
    )
    p_final_ready.add_argument("--no-require-discovery-responsiveness", action="store_true")
    p_final_ready.add_argument(
        "--max-first-candidate-elapsed-s",
        type=float,
        default=final_readiness_thresholds.max_first_candidate_elapsed_s,
    )
    p_final_ready.add_argument("--no-require-closed-loop-state-persistence", action="store_true")
    p_final_ready.add_argument("--no-require-adaptive-live-component-evidence", action="store_true")
    p_final_ready.add_argument("--no-require-target-version-audit", action="store_true")
    p_final_ready.add_argument("--json", action="store_true", help="emit the generated readiness JSON")
    p_final_ready.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="exit with code 2 when any required final-readiness gate is not satisfied",
    )
    p_final_ready.set_defaults(func=handlers.final_readiness)

    p_review_ready = subparsers.add_parser(
        "review-readiness",
        help="audit ISCE-style review readiness from lightweight repository and bug evidence",
    )
    p_review_ready.add_argument("--json", action="store_true", help="emit machine-readable readiness JSON")
    p_review_ready.add_argument("--write-report", action="store_true", help="write reports/review-readiness-*.json and .md")
    p_review_ready.add_argument("--output-dir", default="reports", help="directory for --write-report output")
    p_review_ready.add_argument(
        "--latest-confirmation-file",
        action="append",
        default=[],
        help="latest confirmation JSON file; may be repeated; defaults to experiments/latest_confirmations.json",
    )
    p_review_ready.add_argument("--target-confirmed", type=int, default=20)
    p_review_ready.add_argument("--min-audit-candidates", type=int, default=1)
    p_review_ready.add_argument("--min-discovery-workflows", type=int, default=1)
    p_review_ready.add_argument("--min-generated-issue-drafts", type=int, default=1)
    p_review_ready.add_argument("--min-issue-bundle-families", type=int, default=1)
    p_review_ready.add_argument("--min-pending-issue-drafts", type=int, default=1)
    p_review_ready.add_argument("--min-old-known-issues", type=int, default=1)
    p_review_ready.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="exit with code 2 when any required review gate is not satisfied",
    )
    p_review_ready.set_defaults(func=handlers.review_readiness)
