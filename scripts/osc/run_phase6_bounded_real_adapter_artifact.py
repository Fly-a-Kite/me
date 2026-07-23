#!/usr/bin/env python3
"""Run exactly one source-bound private pandas/polars diagnostic artifact."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc._phase6_gate_authority import verify_source_snapshot  # noqa: E402
from datadiff_osc.runtime._phase6_bounded_real_adapter_artifact import (  # noqa: E402
    SourceSnapshotBinding,
    capture_and_write_bounded_real_adapter_artifact,
    verify_bounded_real_adapter_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-snapshot-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        verified_source = verify_source_snapshot(
            repo_root=args.repo_root,
            snapshot_path=args.source_snapshot,
            expected_snapshot_sha256=args.source_snapshot_sha256,
        )
        source = SourceSnapshotBinding.from_verified(verified_source)
        runner_sha256 = sha256(Path(__file__).read_bytes()).hexdigest()
        artifact, artifact_sha256 = capture_and_write_bounded_real_adapter_artifact(
            source=source,
            runner_sha256=runner_sha256,
            output_path=args.output,
        )
        checked = verify_bounded_real_adapter_artifact(
            artifact_path=args.output,
            expected_sha256=artifact_sha256,
            expected_source=source,
            expected_runner_sha256=runner_sha256,
        )
        if checked != artifact:
            raise ValueError("bounded artifact changed during post-write verification")
        sys.stdout.write(
            canonical_json(
                {
                    "artifact_path": str(args.output.resolve()),
                    "artifact_sha256": artifact_sha256,
                    "artifact_id": artifact.artifact_id,
                    "source_digest": source.source_digest,
                    "diagnostic_only": True,
                    "gate_credit": False,
                    "candidate_confirmed": False,
                    "bug_claimed": False,
                }
            )
            + "\n"
        )
        return 0
    except Exception as exc:
        sys.stderr.write(
            f"bounded real adapter artifact refused: {type(exc).__name__}: {exc}\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
