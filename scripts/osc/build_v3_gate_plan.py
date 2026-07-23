#!/usr/bin/env python3
"""Build the 22-run/2200-case v3 support plan without executing it."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc.runtime.protocols import build_v3_gate_plan  # noqa: E402


def _strings(value: str) -> tuple[str, ...]:
    return tuple(item for item in value.split(",") if item)


def _integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", type=_strings, required=True)
    parser.add_argument("--seeds", type=_integers, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--protocol-digest", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_v3_gate_plan(
        lane_ids=args.lanes,
        seeds=args.seeds,
        source_digest=args.source_digest,
        protocol_digest=args.protocol_digest,
    )
    payload = {
        "plan": plan,
        "planned_runs": plan.planned_runs,
        "planned_cases": plan.planned_cases,
        "plan_digest": plan.digest,
    }
    text = canonical_json(payload) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
