#!/usr/bin/env python3
"""RQ7 — competitor-scope arms over the confirmed-root corpus.

For each confirmed root we ask two questions:
  1. reachability: does the competitor's published backend set contain the buggy
     system at all?
  2. detection: if it does, does a plain differential comparison inside that arm
     actually flag the divergence?

Arms approximate the *scope* (target systems + oracle family) of the strongest
related systems, not their generators:
  - tdiff_style            : DataFrame-API family + plain differential (TDiFf-like)
  - sqlancer_common_scope  : DuckDB/SQLite + SQL-oracle scope (SQLancer-like)
  - ours_full              : all four execution-model families + contract oracle

Usage:
  python scripts/paper/run_rq7_competitor_arms.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from datadiff.backends.version_subprocess import (  # noqa: E402
    compare_normalized,
    run_case_in_environment,
)
from datadiff.version_environments import (  # noqa: E402
    VersionEnvironment,
    probe_package_versions,
)

PROJECT_BACKEND = {
    "apache/datafusion": "datafusion",
    "pola-rs/polars": "polars",
    "apache/arrow": "pyarrow",
    "duckdb/duckdb": "duckdb",
}

ARMS: dict[str, dict[str, object]] = {
    "tdiff_style": {
        "label": "DataFrame family + plain differential (TDiFf-like scope)",
        "backends": ["pandas", "polars", "polars_lazy"],
    },
    "sqlancer_common_scope": {
        "label": "DuckDB/SQLite + SQL-oracle scope (SQLancer-like)",
        "backends": ["duckdb", "sqlite"],
    },
    "ours_full": {
        "label": "all four execution-model families + contract oracle",
        "backends": ["pandas", "polars", "duckdb", "pyarrow", "datafusion", "chdb"],
    },
}


def _current_environment() -> VersionEnvironment:
    return VersionEnvironment(
        env_id="current",
        interpreter=sys.executable,
        packages=dict(probe_package_versions(sys.executable, ("pandas", "polars", "duckdb", "pyarrow", "datafusion"))),
        label="current interpreter",
        repository_root=str(REPO_ROOT),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "paper/experiments/results")
    args = parser.parse_args()

    corpus = json.loads(
        (REPO_ROOT / "experiments/canonical_confirmed_bug_corpus/v2/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    environment = _current_environment()

    rows = []
    for root in corpus["confirmed_roots"]:
        project = root["project"]
        buggy = PROJECT_BACKEND.get(project)
        case_path = REPO_ROOT / root["dsl_case"]["path"]
        case = json.loads(case_path.read_text(encoding="utf-8"))
        per_arm: dict[str, object] = {}
        for arm_name, arm in ARMS.items():
            backends = list(arm["backends"])  # type: ignore[arg-type]
            if buggy not in backends:
                per_arm[arm_name] = {"reachable": False, "detected": False, "reason": "buggy backend out of scope"}
                continue
            reference = next((b for b in backends if b != buggy), None)
            if reference is None:
                per_arm[arm_name] = {"reachable": True, "detected": False, "reason": "no reference backend"}
                continue
            left = run_case_in_environment(environment, case, buggy, timeout_s=args.timeout_s)
            right = run_case_in_environment(environment, case, reference, timeout_s=args.timeout_s)
            comparison = compare_normalized(left, right) if left.get("status") == "ok" and right.get("status") == "ok" else {
                "match": left.get("status") == right.get("status"),
                "reason": "status",
            }
            per_arm[arm_name] = {
                "reachable": True,
                "detected": not comparison["match"],
                "reason": comparison["reason"],
                "buggy_status": left.get("status"),
                "reference_status": right.get("status"),
                "reference_backend": reference,
                "buggy_rows": (left.get("normalized") or {}).get("rows"),
                "reference_rows": (right.get("normalized") or {}).get("rows"),
            }
        rows.append(
            {
                "root_id": root["root_id"],
                "project": project,
                "families": root.get("observed_families", []),
                "buggy_backend": buggy,
                "arms": per_arm,
            }
        )

    reachable = {arm: sum(1 for r in rows if r["arms"][arm]["reachable"]) for arm in ARMS}
    detected = {arm: sum(1 for r in rows if r["arms"][arm]["detected"]) for arm in ARMS}
    union_reachable = sum(
        1 for r in rows if any(r["arms"][arm]["reachable"] for arm in ARMS if arm != "ours_full")
    )

    payload = {
        "schema_version": "datadiff-rq7-competitor-arms-v1",
        "confirmed_roots": len(rows),
        "arms": ARMS,
        "reachable_counts": reachable,
        "detected_counts": detected,
        "competitor_union_reachable": union_reachable,
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "rq7_competitor_arms.json"
    md_path = args.out_dir / "rq7_competitor_arms.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# RQ7 — competitor-scope arm results",
        "",
        f"- confirmed roots: {len(rows)}",
        f"- reachable (buggy backend in scope): {reachable}",
        f"- detected (plain differential flagged it): {detected}",
        f"- competitor union reachable: {union_reachable}/{len(rows)} "
        f"→ {len(rows) - union_reachable}/{len(rows)} out of reach for all competitor scopes",
        "",
        "| root | buggy backend | tdiff reach | tdiff detect | sqlancer reach | sqlancer detect | ours reach | ours detect |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        a = r["arms"]
        lines.append(
            f"| {r['root_id']} | {r['buggy_backend']} | "
            f"{a['tdiff_style']['reachable']} | {a['tdiff_style']['detected']} | "
            f"{a['sqlancer_common_scope']['reachable']} | {a['sqlancer_common_scope']['detected']} | "
            f"{a['ours_full']['reachable']} | {a['ours_full']['detected']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"reachable={reachable} detected={detected} union_reachable={union_reachable}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
