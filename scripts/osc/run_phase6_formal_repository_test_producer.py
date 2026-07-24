#!/usr/bin/env python3
"""Run a separately authorized Phase-6 repository-test receipt producer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc.runtime._phase6_formal_repository_test_producer import (  # noqa: E402
    execute_formal_repository_test,
    formal_repository_test_producer_preview,
    load_formal_repository_test_execution_authority,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authority",
        type=Path,
        help="exact future repository-test execution authority document",
    )
    parser.add_argument(
        "--authority-sha256",
        help="out-of-band SHA-256 for --authority",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute only an exact separately authorized repository-test workload",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        if args.authority is not None or args.authority_sha256 is not None:
            _parser().error("--authority and --authority-sha256 require --execute")
        sys.stdout.write(canonical_json(formal_repository_test_producer_preview()) + "\n")
        return 0
    if args.authority is None or not args.authority_sha256:
        _parser().error("--execute requires --authority and --authority-sha256")
    authority = load_formal_repository_test_execution_authority(
        path=args.authority,
        expected_sha256=args.authority_sha256,
    )
    result = execute_formal_repository_test(authority=authority)
    sys.stdout.write(canonical_json(result.to_dict()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
