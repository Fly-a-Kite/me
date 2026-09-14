#!/usr/bin/env python3
"""Parallel-invariance scan: same large-table case on datafusion (1 partition) vs
datafusion_parallel (N partitions).

Small corpus tables never partition, so this scales fresh cases to a large row
count first. A mismatch is a parallelism-sensitive correctness difference.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from datadiff.backends.version_subprocess import (  # noqa: E402
    compare_normalized,
    run_case_in_environment,
)
from datadiff.datagen import generate_case  # noqa: E402
from datadiff.version_environments import VersionEnvironment  # noqa: E402


def _scale_rows(case: dict, rows: int, seed: int) -> dict:
    """Bootstrap-resample every table to ``rows`` rows, keeping schema and program."""
    rnd = random.Random(seed)
    scaled = copy.deepcopy(case)
    for table in scaled.get("tables", []) or []:
        existing = table.get("rows") or []
        if existing:
            table["rows"] = [dict(rnd.choice(existing)) for _ in range(rows)]
    return scaled


def _float_only_difference(left_rows, right_rows, rel_tol: float = 1e-9) -> bool:
    """True when aligned rows differ only by float reduction-order rounding."""
    if not isinstance(left_rows, list) or not isinstance(right_rows, list):
        return False
    if len(left_rows) != len(right_rows):
        return False
    for left_row, right_row in zip(left_rows, right_rows):
        if len(left_row) != len(right_row):
            return False
        for a, b in zip(left_row, right_row):
            numeric = (
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and not isinstance(a, bool)
                and not isinstance(b, bool)
            )
            if numeric:
                if abs(a - b) > rel_tol * max(1.0, abs(a), abs(b)):
                    return False
            elif a != b:
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="31200101,31200102,31200103,31200104,31200105")
    parser.add_argument("--profile", default="discovery_fresh")
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--left-backend", default="datafusion")
    parser.add_argument("--right-backend", default="datafusion_parallel")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "paper/experiments/results/parallel_invariance"
    )
    args = parser.parse_args()

    environment = VersionEnvironment(
        env_id="current", interpreter=sys.executable, repository_root=str(REPO_ROOT)
    )
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    results = []
    for seed in seeds:
        case = _scale_rows(generate_case(seed, profile=args.profile).to_dict(), args.rows, seed)
        left = run_case_in_environment(
            environment, case, args.left_backend, timeout_s=args.timeout_s
        )
        right = run_case_in_environment(
            environment, case, args.right_backend, timeout_s=args.timeout_s
        )
        comparison = compare_normalized(left, right)
        left_rows = (left.get("normalized") or {}).get("rows")
        right_rows = (right.get("normalized") or {}).get("rows")
        kind = "match"
        if not comparison["match"]:
            kind = "result"
            if (
                left.get("status") == "ok"
                and right.get("status") == "ok"
                and _float_only_difference(left_rows, right_rows)
            ):
                kind = "expected_float_reduction_order"
        results.append(
            {
                "seed": seed,
                "left_status": left.get("status"),
                "right_status": right.get("status"),
                "match": comparison["match"],
                "reason": comparison["reason"],
                "kind": kind,
                "left_rows": left_rows,
                "right_rows": right_rows,
            }
        )

    findings = [r for r in results if not r["match"]]
    real_findings = [r for r in findings if r["kind"] != "expected_float_reduction_order"]
    payload = {
        "schema_version": "datadiff-parallel-invariance-v1",
        "profile": args.profile,
        "rows": args.rows,
        "left_backend": args.left_backend,
        "right_backend": args.right_backend,
        "cases": len(results),
        "findings": len(findings),
        "real_findings": len(real_findings),
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "parallel_invariance.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"cases={len(results)} findings={len(findings)} real_findings={len(real_findings)}")
    for finding in real_findings[:10]:
        print(f"  REAL finding seed={finding['seed']} reason={finding['reason']}")
    print(f"json={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
