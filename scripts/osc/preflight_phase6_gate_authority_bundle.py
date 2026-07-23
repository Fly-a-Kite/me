#!/usr/bin/env python3
"""Recompute the Root/Runtime Phase-6 gate chain without authorizing 24h."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc._phase6_gate_authority import (  # noqa: E402
    preflight_phase6_gate_authority_bundle,
    verify_artifact_receipt_index,
    verify_dynamic_gate_plan,
    verify_source_snapshot,
)
from datadiff_osc.semantic_targets.declarations import (  # noqa: E402
    legacy_v4_target_templates,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--dynamic-plan", type=Path, required=True)
    parser.add_argument("--dynamic-plan-sha256", required=True)
    parser.add_argument("--artifact-receipts", type=Path, required=True)
    parser.add_argument("--artifact-receipts-sha256", required=True)
    parser.add_argument("--authority-plan", type=Path, required=True)
    parser.add_argument("--authority-plan-sha256", required=True)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--artifact-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = verify_source_snapshot(
            repo_root=args.repo_root,
            snapshot_path=args.source_snapshot,
            expected_snapshot_sha256=args.source_snapshot_sha256,
        )
        dynamic_plan = verify_dynamic_gate_plan(
            plan_path=args.dynamic_plan,
            expected_plan_sha256=args.dynamic_plan_sha256,
            expected_source_digest=source.source_digest,
        )
        receipts = verify_artifact_receipt_index(
            index_path=args.artifact_receipts,
            expected_index_sha256=args.artifact_receipts_sha256,
            expected_source_digest=source.source_digest,
            expected_dynamic_plan_sha256=dynamic_plan.byte_sha256,
        )
        report = preflight_phase6_gate_authority_bundle(
            source=source,
            templates=legacy_v4_target_templates(),
            dynamic_plan=dynamic_plan,
            artifact_receipts=receipts,
            authority_plan_path=args.authority_plan,
            expected_authority_plan_sha256=args.authority_plan_sha256,
            artifact_manifest_path=args.artifact_manifest,
            expected_artifact_manifest_sha256=args.artifact_manifest_sha256,
        )
        text = canonical_json(report.to_dict()) + "\n"
        sys.stdout.write(text)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        return 0 if report.phase6_gate_eligible else 2
    except Exception as exc:  # every malformed or racing input is a no-go
        sys.stderr.write(f"phase6 authority preflight refused: {type(exc).__name__}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
