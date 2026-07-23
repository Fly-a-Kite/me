#!/usr/bin/env python3
"""Build a Root Phase-6 authority plan, or fail closed without an output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc._phase6_gate_authority import (  # noqa: E402
    build_phase6_gate_authority_bundle,
    publish_phase6_gate_authority_bundle,
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _quarantine_broken_pipe(stream: object) -> None:
    """Prevent interpreter shutdown from retrying a failed buffered pipe."""

    try:
        stream_fd = stream.fileno()  # type: ignore[attr-defined]
    except (AttributeError, OSError, ValueError):
        return
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        return
    try:
        os.dup2(devnull_fd, stream_fd)
    except OSError:
        pass
    finally:
        if devnull_fd != stream_fd:
            os.close(devnull_fd)


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
        templates = legacy_v4_target_templates()
        result = build_phase6_gate_authority_bundle(
            source=source,
            templates=templates,
            dynamic_plan=dynamic_plan,
            artifact_receipts=receipts,
        )
        sys.stdout.write(canonical_json(result.to_dict()) + "\n")
        sys.stdout.flush()
        if not result.publishable:
            return 2
        publish_phase6_gate_authority_bundle(
            result=result,
            source=source,
            templates=templates,
            dynamic_plan=dynamic_plan,
            artifact_receipts=receipts,
            output_path=args.output,
        )
        return 0
    except Exception as exc:  # every malformed or racing input is a no-go
        if isinstance(exc, BrokenPipeError):
            _quarantine_broken_pipe(sys.stdout)
        try:
            sys.stderr.write(
                f"phase6 authority build refused: {type(exc).__name__}: {exc}\n"
            )
            sys.stderr.flush()
        except BrokenPipeError:
            _quarantine_broken_pipe(sys.stderr)
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
