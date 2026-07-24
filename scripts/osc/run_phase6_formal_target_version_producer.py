#!/usr/bin/env python3
"""Print the private Phase-6 target-version preparation boundary only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc.runtime._phase6_formal_target_version_producer import (  # noqa: E402
    target_version_producer_preview,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the fixed preparation-only declaration (the default)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _parser().parse_args(argv)
    sys.stdout.write(canonical_json(target_version_producer_preview()) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
