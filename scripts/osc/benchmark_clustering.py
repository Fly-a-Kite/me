#!/usr/bin/env python3
"""Benchmark linear fingerprint clustering against an explicit pairwise oracle."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from datadiff_osc._canonical import canonical_json  # noqa: E402
from datadiff_osc.contract_engine.fingerprints import ComponentFingerprint  # noqa: E402
from datadiff_osc.runtime.benchmarks import benchmark_component_clustering  # noqa: E402


def _parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item)
    if not sizes or any(item < 1 for item in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def _group(size: int) -> tuple[ComponentFingerprint, ...]:
    return tuple(
        ComponentFingerprint(
            endpoint_id=f"endpoint-{index:05d}",
            observer_id="bag",
            observer_digest="observer-v1",
            contract_digest="synthetic-contract-v1",
            row_count=32,
            schema_digest="schema-v1",
            payload_digest=f"payload-{index % max(1, size // 3)}",
        )
        for index in range(size)
    )


def _emit(payload, output: Path | None) -> None:
    text = canonical_json(payload) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-counts", type=_parse_sizes, default=(2, 4, 8, 16, 32))
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark_component_clustering(
        tuple(_group(size) for size in args.backend_counts),
        repetitions=args.repetitions,
    )
    _emit(
        {
            "schema_version": "osc-clustering-benchmark-script-output-v1",
            "report": report,
            "twenty_four_hour_run_authorized": False,
        },
        args.output,
    )
    return 0 if report.all_pairwise_partitions_match else 2


if __name__ == "__main__":
    raise SystemExit(main())
