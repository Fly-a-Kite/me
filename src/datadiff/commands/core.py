from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


CommandHandler = Callable[[argparse.Namespace], int]


@dataclass(frozen=True, slots=True)
class CoreCommandHandlers:
    init: CommandHandler
    targets: CommandHandler
    semantic_registry: CommandHandler
    target_version_audit: CommandHandler
    version_ledger: CommandHandler
    prune_corpus: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: CoreCommandHandlers,
    add_target_suite_flags: Callable[[argparse.ArgumentParser], None],
) -> None:
    p_init = subparsers.add_parser("init", help="create project runtime directories")
    p_init.set_defaults(func=handlers.init)

    p_targets = subparsers.add_parser("targets", help="list supported backend targets and target suites")
    p_targets.add_argument("--json", action="store_true", help="emit target registry as JSON")
    p_targets.set_defaults(func=handlers.targets)

    p_semantic_registry = subparsers.add_parser(
        "semantic-registry",
        help="emit the reusable semantic objective/capability/oracle methodology registry",
    )
    add_target_suite_flags(p_semantic_registry)
    p_semantic_registry.add_argument(
        "--exploration-objective-rules",
        default="",
        help="JSON or @path defining extra neutral exploration objective rules",
    )
    p_semantic_registry.add_argument("--json", action="store_true", help="emit registry as JSON")
    p_semantic_registry.set_defaults(func=handlers.semantic_registry)

    p_target_version_audit = subparsers.add_parser(
        "target-version-audit",
        help="audit installed target backend package versions against latest package versions",
    )
    p_target_version_audit.add_argument(
        "--packages",
        default="",
        help="comma-separated PyPI packages to audit; defaults to registered real target packages",
    )
    p_target_version_audit.add_argument(
        "--latest-versions",
        default="",
        help="JSON object, @json-file, or comma-separated package=version overrides for offline audits",
    )
    p_target_version_audit.add_argument(
        "--no-network",
        action="store_true",
        help="do not query PyPI; latest versions must come from --latest-versions",
    )
    p_target_version_audit.add_argument("--output", default="", help="optional audit JSON output path")
    p_target_version_audit.add_argument("--json", action="store_true", help="emit audit as JSON")
    p_target_version_audit.set_defaults(func=handlers.target_version_audit)

    p_version_ledger = subparsers.add_parser(
        "version-ledger",
        help="build a cross-version candidate-family ledger from one or more run logs",
    )
    p_version_ledger.add_argument("--run-file", default=None, help="single run log; defaults to latest run")
    p_version_ledger.add_argument("--run-files", default="", help="comma-separated run logs in version order")
    p_version_ledger.add_argument(
        "--manifest-index",
        action="append",
        default=[],
        help=(
            "final experiment manifest index to scan for run logs when --run-files is omitted; "
            "may be repeated"
        ),
    )
    p_version_ledger.add_argument("--versions", default="", help="comma-separated version ids matching --run-files")
    p_version_ledger.add_argument("--baseline-version", default="", help="baseline version id; defaults to first run")
    p_version_ledger.add_argument("--previous-ledger", default="", help="previous ledger JSON for regression detection")
    p_version_ledger.add_argument("--output", default="", help="optional ledger JSON output path")
    p_version_ledger.add_argument(
        "--evidence-manifest-output",
        default="",
        help="optional final-readiness evidence manifest that references the written ledger",
    )
    p_version_ledger.add_argument("--json", action="store_true", help="emit ledger as JSON")
    p_version_ledger.set_defaults(func=handlers.version_ledger)

    p_prune = subparsers.add_parser("prune-corpus", help="dry-run prune of persisted feedback corpus cases")
    p_prune.add_argument(
        "--keep",
        type=int,
        default=4096,
        help="number of newest corpus/interesting JSON files to keep",
    )
    p_prune.add_argument("--yes", action="store_true", help="delete files beyond --keep")
    p_prune.set_defaults(func=handlers.prune_corpus)
