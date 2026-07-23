#!/usr/bin/env python3
"""Calculate a fail-closed OSC gate report from verified raw artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc.runtime.gate_artifacts import (  # noqa: E402
    verify_gate_authority_plan,
    verify_gate_artifact_manifest,
)
from datadiff_osc.runtime.gates import calculate_gate_report  # noqa: E402


def _sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("expected a lowercase SHA-256")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--authority-plan", type=Path, required=True)
    parser.add_argument("--authority-plan-sha256", type=_sha256, required=True)
    parser.add_argument("--expected-source-digest", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    authority = verify_gate_authority_plan(
        args.authority_plan,
        expected_sha256=args.authority_plan_sha256,
        expected_source_digest=args.expected_source_digest,
    )
    verified = verify_gate_artifact_manifest(
        args.input,
        expected_source_digest=args.expected_source_digest,
        expected_authority_plan_sha256=args.authority_plan_sha256,
        authority_plan=authority,
    )
    report = calculate_gate_report(
        verified,
        expected_source_digest=args.expected_source_digest,
        expected_authority_plan_sha256=args.authority_plan_sha256,
        authority_plan=authority,
    )
    text = canonical_json(report) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if report.all_gates_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
