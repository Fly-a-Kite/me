#!/usr/bin/env python3
"""Build a non-executing preregistered 2h paired-pilot plan."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc.runtime.protocols import build_paired_pilot_plan  # noqa: E402


def _integers(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(",") if item)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed blocks must be integers") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--seed-blocks", type=_integers, required=True)
    parser.add_argument("--case-cap-per-arm", type=int, required=True)
    parser.add_argument("--cpu-seconds-cap-per-arm", type=int, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--environment-digest", required=True)
    parser.add_argument("--target-digest", required=True)
    parser.add_argument("--contract-digest", required=True)
    parser.add_argument("--phase6-gate-report-digest", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build_paired_pilot_plan(
        treatment_id=args.treatment,
        control_id=args.control,
        seed_blocks=args.seed_blocks,
        case_cap_per_arm=args.case_cap_per_arm,
        cpu_seconds_cap_per_arm=args.cpu_seconds_cap_per_arm,
        source_digest=args.source_digest,
        environment_digest=args.environment_digest,
        target_digest=args.target_digest,
        contract_digest=args.contract_digest,
        phase6_gate_report_digest=args.phase6_gate_report_digest,
    )
    text = canonical_json({"plan": plan, "plan_digest": plan.digest}) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
