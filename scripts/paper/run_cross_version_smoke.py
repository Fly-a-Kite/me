#!/usr/bin/env python3
"""Run a cross-version differential smoke over the canonical corpus cases.

Example:
  python scripts/paper/run_cross_version_smoke.py \
      --left datafusion-53.0.0--pyarrow-24.0.0 \
      --right frozen-target \
      --backend datafusion
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from datadiff.backends.version_subprocess import cross_version_compare  # noqa: E402
from datadiff.version_environments import load_registry  # noqa: E402

_BACKEND_BY_PREFIX = {
    "datafusion": "datafusion",
    "polars": "polars",
    "pyarrow": "pyarrow",
    "duckdb": "duckdb",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="left env_id (e.g. the older version)")
    parser.add_argument("--right", required=True, help="right env_id (e.g. the newer version)")
    parser.add_argument("--backend", default="", help="restrict to one backend; default: infer from case name")
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "paper/experiments/results"
    )
    args = parser.parse_args()

    registry = load_registry()
    left = registry[args.left]
    right = registry[args.right]
    cases = sorted(glob.glob(str(REPO_ROOT / "experiments/canonical_confirmed_bug_corpus/v1/cases/*.json")))

    results = []
    for path in cases:
        name = os.path.basename(path)
        backend = args.backend or _BACKEND_BY_PREFIX.get(name.split("_")[0], "")
        if not backend:
            continue
        case = json.loads(Path(path).read_text(encoding="utf-8"))
        outcome = cross_version_compare(left, right, case, backend, timeout_s=args.timeout_s)
        results.append(
            {
                "case": name,
                "backend": backend,
                "left_status": outcome["left"].get("status"),
                "right_status": outcome["right"].get("status"),
                "comparison": outcome["comparison"],
                "left_rows": (outcome["left"].get("normalized") or {}).get("rows"),
                "right_rows": (outcome["right"].get("normalized") or {}).get("rows"),
            }
        )

    mismatches = [r for r in results if not r["comparison"]["match"]]
    payload = {
        "schema_version": "datadiff-cross-version-smoke-v1",
        "left_env_id": args.left,
        "right_env_id": args.right,
        "left_packages": dict(sorted(left.packages.items())),
        "right_packages": dict(sorted(right.packages.items())),
        "cases": len(results),
        "mismatches": len(mismatches),
        "results": results,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"cross_version_{args.left}__{args.right}.json"
    md_path = args.out_dir / f"cross_version_{args.left}__{args.right}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# Cross-version smoke: `{args.left}` vs `{args.right}`",
        "",
        f"- left packages: {payload['left_packages']}",
        f"- right packages: {payload['right_packages']}",
        f"- cases: {len(results)}; mismatches: {len(mismatches)}",
        "",
        "| case | backend | left | right | match | reason |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r['case']} | {r['backend']} | {r['left_status']} | {r['right_status']} | "
            f"{r['comparison']['match']} | {r['comparison']['reason']} |"
        )
    lines.append("")
    for r in mismatches:
        lines.append(f"## Mismatch: {r['case']}")
        lines.append("")
        lines.append(f"- left rows: `{json.dumps(r['left_rows'], ensure_ascii=False)}`")
        lines.append(f"- right rows: `{json.dumps(r['right_rows'], ensure_ascii=False)}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"cases={len(results)} mismatches={len(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
