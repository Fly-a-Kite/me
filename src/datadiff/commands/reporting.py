from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class ReportingCommandHandlers:
    report: CommandHandler
    bug_audit: CommandHandler
    bug_status: CommandHandler
    issue_readiness: CommandHandler
    issue_bundle: CommandHandler
    methodology_report: CommandHandler
    show_bugs: CommandHandler
    classify_run: CommandHandler
    run_health: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: ReportingCommandHandlers,
    list_audit_probe_ids: Callable[[], Sequence[str]],
    default_issue_bundle_statuses: Sequence[str],
) -> None:
    p_report = subparsers.add_parser("report", help="generate markdown/csv report")
    p_report.add_argument("--run-file", default=None)
    p_report.add_argument(
        "--csv-limit",
        type=int,
        default=None,
        help="maximum finding rows to export to CSV; useful for large longrun logs",
    )
    p_report.set_defaults(func=handlers.report)

    p_bug_audit = subparsers.add_parser(
        "bug-audit",
        help="run deterministic latest-version bug probes and write a machine-readable evidence manifest",
    )
    p_bug_audit.add_argument(
        "--probes",
        default="",
        help=f"comma-separated probe ids; defaults to all: {','.join(list_audit_probe_ids())}",
    )
    p_bug_audit.add_argument(
        "--fail-on-candidate",
        action="store_true",
        help="exit with code 2 when any candidate implementation bug is detected",
    )
    p_bug_audit.add_argument(
        "--write-issues",
        action="store_true",
        help="write candidate issue drafts into --issue-dir using the automated audit manifest",
    )
    p_bug_audit.add_argument(
        "--issue-dir",
        default="new_issue/generated",
        help="directory for --write-issues output; defaults to new_issue/generated",
    )
    p_bug_audit.add_argument(
        "--overwrite-issues",
        action="store_true",
        help="overwrite existing audit-generated issue drafts",
    )
    p_bug_audit.set_defaults(func=handlers.bug_audit)

    p_bug_status = subparsers.add_parser(
        "bug-status",
        help="summarize confirmed, candidate, generated, and old-known bug evidence without scanning run logs",
    )
    p_bug_status.add_argument("--json", action="store_true", help="emit machine-readable status JSON")
    p_bug_status.add_argument(
        "--latest-confirmations",
        default="",
        help="comma-separated latest confirmation JSON files; defaults to experiments/latest_confirmations.json",
    )
    p_bug_status.add_argument("--new-issue-dir", default="new_issue")
    p_bug_status.add_argument("--old-issue-dir", default="old_issue")
    p_bug_status.add_argument("--generated-issue-dir", default="new_issue/generated")
    p_bug_status.add_argument("--write-report", action="store_true", help="write reports/bug-status-*.json and .md")
    p_bug_status.add_argument("--output-dir", default="reports", help="directory for --write-report output")
    p_bug_status.set_defaults(func=handlers.bug_status)

    p_issue_readiness = subparsers.add_parser(
        "issue-readiness",
        help="audit local issue drafts for submission readiness without scanning run logs",
    )
    p_issue_readiness.add_argument("--json", action="store_true", help="emit machine-readable readiness JSON")
    p_issue_readiness.add_argument(
        "--latest-confirmations",
        default="",
        help="comma-separated latest confirmation JSON files; defaults to experiments/latest_confirmations.json",
    )
    p_issue_readiness.add_argument("--new-issue-dir", default="new_issue")
    p_issue_readiness.add_argument("--old-issue-dir", default="old_issue")
    p_issue_readiness.add_argument("--generated-issue-dir", default="new_issue/generated")
    p_issue_readiness.add_argument(
        "--include-generated",
        action="store_true",
        help="also audit raw generated issue drafts under --generated-issue-dir",
    )
    p_issue_readiness.add_argument(
        "--write-report",
        action="store_true",
        help="write reports/issue-readiness-*.json and .md",
    )
    p_issue_readiness.add_argument("--output-dir", default="reports", help="directory for --write-report output")
    p_issue_readiness.add_argument(
        "--fail-on-no-ready",
        action="store_true",
        help="exit with code 2 when no local issue draft is ready to submit",
    )
    p_issue_readiness.set_defaults(func=handlers.issue_readiness)

    p_issue_bundle = subparsers.add_parser(
        "issue-bundle",
        help="extract local issue reproducers and write a portable evidence manifest",
    )
    p_issue_bundle.add_argument("--json", action="store_true", help="emit the generated bundle manifest as JSON")
    p_issue_bundle.add_argument(
        "--latest-confirmations",
        default="",
        help="comma-separated latest confirmation JSON files; defaults to experiments/latest_confirmations.json",
    )
    p_issue_bundle.add_argument("--new-issue-dir", default="new_issue")
    p_issue_bundle.add_argument("--old-issue-dir", default="old_issue")
    p_issue_bundle.add_argument("--generated-issue-dir", default="new_issue/generated")
    p_issue_bundle.add_argument(
        "--statuses",
        default=",".join(default_issue_bundle_statuses),
        help="comma-separated issue-readiness statuses to bundle",
    )
    p_issue_bundle.add_argument(
        "--output-dir",
        default="new_issue/generated/issue-bundles",
        help="directory for manifest and extracted reproducers",
    )
    p_issue_bundle.add_argument(
        "--run-reproducers",
        action="store_true",
        help="execute extracted reproducers and capture stdout/stderr in the manifest",
    )
    p_issue_bundle.add_argument("--timeout", type=float, default=20.0, help="per-reproducer timeout in seconds")
    p_issue_bundle.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="number of times to execute each reproducer when --run-reproducers is set",
    )
    p_issue_bundle.add_argument(
        "--primary-per-family",
        action="store_true",
        help="bundle only the primary selected issue draft for each issue-readiness submission family",
    )
    p_issue_bundle.add_argument(
        "--fail-on-missing-reproducer",
        action="store_true",
        help="exit with code 2 if any selected issue lacks a Python reproducer block",
    )
    p_issue_bundle.add_argument(
        "--fail-on-compile-error",
        action="store_true",
        help="exit with code 2 if any extracted reproducer has a syntax error",
    )
    p_issue_bundle.set_defaults(func=handlers.issue_bundle)

    p_methodology_report = subparsers.add_parser(
        "methodology-report",
        help="write a paper-facing methodology report from an experiment manifest",
    )
    p_methodology_report.add_argument("--manifest", default=None)
    p_methodology_report.add_argument(
        "--refresh",
        action="store_true",
        help="recompute experiment summary findings with the current oracle before reporting",
    )
    p_methodology_report.add_argument(
        "--summary-only",
        action="store_true",
        help="reuse existing experiment-summary CSVs when available and skip run-log-derived report sections",
    )
    p_methodology_report.add_argument("--json", action="store_true", help="emit the generated report JSON")
    p_methodology_report.set_defaults(func=handlers.methodology_report)

    p_show = subparsers.add_parser("show-bugs", help="print bug findings")
    p_show.add_argument("--run-file", default=None)
    p_show.add_argument("--limit", type=int, default=10)
    p_show.set_defaults(func=handlers.show_bugs)

    p_classify = subparsers.add_parser("classify-run", help="classify findings as bugs, semantic divergences, or false positives")
    p_classify.add_argument("--run-file", default=None)
    p_classify.add_argument("--limit", type=int, default=3, help="examples per verdict")
    p_classify.add_argument(
        "--refresh",
        action="store_true",
        help="recompute differential findings from stored normalized outputs with the current oracle",
    )
    p_classify.add_argument("--json", action="store_true", help="emit machine-readable classification summary")
    p_classify.set_defaults(func=handlers.classify_run)

    p_health = subparsers.add_parser("run-health", help="summarize an in-progress or completed run log")
    p_health.add_argument("--run-file", default=None)
    p_health.add_argument("--limit", type=int, default=3, help="candidate examples to show")
    p_health.add_argument("--json", action="store_true", help="emit machine-readable health summary")
    p_health.add_argument(
        "--fail-on-fresh-candidate",
        action="store_true",
        help="exit with code 2 when the run contains an unsaturated organic candidate family",
    )
    p_health.add_argument("--fail-on-bug", action="store_true", help="exit with code 2 when any row has status=bug")
    p_health.set_defaults(func=handlers.run_health)
