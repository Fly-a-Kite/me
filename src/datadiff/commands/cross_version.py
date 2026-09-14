"""CLI command for the cross-version discovery lane."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datadiff.cross_version_scan import DEFAULT_BACKENDS, scan_cross_version, write_scan
from datadiff.version_environments import load_registry, repository_root

CommandHandler = Callable[[argparse.Namespace], int]


def _parse_pairs(args: argparse.Namespace) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for raw in list(getattr(args, "env_pair", []) or []):
        text = str(raw).strip()
        if not text:
            continue
        if "->" not in text:
            raise SystemExit(f"--env-pair must be LEFT->RIGHT, got {text!r}")
        left, right = (part.strip() for part in text.split("->", 1))
        pairs.append((left, right))
    left = str(getattr(args, "left_env", "") or "").strip()
    right = str(getattr(args, "right_env", "") or "").strip()
    if left or right:
        if not (left and right):
            raise SystemExit("--left-env and --right-env must be provided together")
        pairs.append((left, right))
    if not pairs:
        raise SystemExit("provide --env-pair LEFT->RIGHT or --left-env/--right-env")
    return pairs


def cmd_cross_version_scan(args: argparse.Namespace) -> int:
    registry = load_registry(getattr(args, "registry", "") or None)
    if not registry:
        raise SystemExit(
            "no version environments registered; run "
            "`python -m datadiff.version_environments --interpreter ... --env-id ... --out experiments/version_environments.json`"
        )
    pairs = _parse_pairs(args)
    backends = [
        item.strip() for item in str(getattr(args, "backends", "") or "").split(",") if item.strip()
    ] or list(DEFAULT_BACKENDS)
    cases = str(getattr(args, "cases", "") or "").strip()
    if not cases:
        cases = str(repository_root() / "experiments/canonical_confirmed_bug_corpus/v1/cases")
    timeout_s = float(getattr(args, "timeout_s", 30.0))

    payload = scan_cross_version(
        registry,
        pairs,
        cases=cases,
        backends=backends,
        timeout_s=timeout_s,
    )
    out_dir = str(getattr(args, "output_dir", "") or "").strip() or str(
        repository_root() / "paper/experiments/results/cross_version_scan"
    )
    json_path, md_path = write_scan(payload, out_dir)

    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"env_pairs={payload['env_pairs']}")
    print(f"backends={payload['backends']}")
    print(f"results={payload['result_count']}")
    print(f"findings={payload['finding_count']}")
    for finding in payload["findings"]:
        print(
            f"  finding: {finding['case']} / {finding['backend']} "
            f"({finding['left_env_id']} vs {finding['right_env_id']})"
        )
    print(f"json={json_path}")
    print(f"markdown={md_path}")
    return 0


@dataclass(frozen=True, slots=True)
class CrossVersionCommandHandlers:
    scan: CommandHandler


def register(
    subparsers: Any,
    *,
    handlers: CrossVersionCommandHandlers | None = None,
) -> None:
    scan_handler = handlers.scan if handlers is not None else cmd_cross_version_scan
    parser = subparsers.add_parser(
        "cross-version-scan",
        help="run the same cases across pinned version environments and report divergences",
    )
    parser.add_argument("--left-env", default="", help="left (e.g. older) environment id")
    parser.add_argument("--right-env", default="", help="right (e.g. newer) environment id")
    parser.add_argument(
        "--env-pair",
        action="append",
        default=[],
        help="LEFT->RIGHT environment pair; may be repeated",
    )
    parser.add_argument(
        "--backends",
        default="",
        help="comma-separated backends; defaults to datafusion,polars,duckdb,pyarrow,pandas",
    )
    parser.add_argument("--cases", default="", help="case JSON file or directory; defaults to the canonical corpus")
    parser.add_argument("--registry", default="", help="version environment registry JSON path")
    parser.add_argument("--output-dir", default="", help="output directory for the scan report")
    parser.add_argument("--timeout-s", type=float, default=30.0, help="per-execution timeout")
    parser.add_argument("--json", action="store_true", help="emit the scan payload as JSON")
    parser.set_defaults(func=scan_handler)
