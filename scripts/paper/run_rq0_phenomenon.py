#!/usr/bin/env python3
"""RQ0 — cross-engine semantic divergence: phenomenon measurement.

Streams campaign run logs and reports how often the same typed workflow produces
divergent results across backends, plus the root-cause / mismatch-class taxonomy.

Example:
  python scripts/paper/run_rq0_phenomenon.py \
      --manifest-glob 'new_issue/generated/discovery-campaign-longhaul-*.json' \
      --limit 300
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_lane_map(manifest_glob: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in sorted(REPO_ROOT.glob(manifest_glob)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for run in payload.get("runs", []) or []:
            run_file = str(run.get("run_file", "") or "")
            if run_file:
                mapping[run_file] = str(run.get("lane_id", "") or "")
    return mapping


def _iter_cases(run_file: Path):
    try:
        with gzip.open(run_file, "rt", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (OSError, EOFError):
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-glob",
        default="new_issue/generated/discovery-campaign-longhaul-20260914T140327Z-batch*.json",
    )
    parser.add_argument("--limit", type=int, default=300, help="max run logs to scan")
    parser.add_argument("--seed", type=int, default=0, help="deterministic sample seed")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "paper/experiments/results/rq0/rq0_phenomenon.json"
    )
    args = parser.parse_args()

    lane_of = _run_lane_map(args.manifest_glob)
    run_files = sorted((REPO_ROOT / name).resolve() for name in lane_of)
    if args.seed:
        import random

        random.Random(args.seed).shuffle(run_files)
    run_files = run_files[: args.limit]

    cases = 0
    cases_with_divergence = 0
    findings = 0
    by_root: Counter[str] = Counter()
    by_mismatch: Counter[str] = Counter()
    by_backend: Counter[str] = Counter()
    by_oracle: Counter[str] = Counter()
    by_lane_cases: Counter[str] = Counter()
    by_lane_divergent: Counter[str] = Counter()

    for run_file in run_files:
        lane = lane_of.get(str(run_file), "")
        for record in _iter_cases(run_file):
            cases += 1
            by_lane_cases[lane] += 1
            case_findings = record.get("findings") or []
            if case_findings:
                cases_with_divergence += 1
                by_lane_divergent[lane] += 1
            for finding in case_findings:
                findings += 1
                by_root[str(finding.get("root_cause", ""))] += 1
                by_mismatch[str(finding.get("mismatch_class", ""))] += 1
                by_oracle[str(finding.get("oracle", ""))] += 1
                for backend in finding.get("suspicious_backends", []) or []:
                    by_backend[str(backend)] += 1

    rate = (cases_with_divergence / cases) if cases else 0.0
    payload = {
        "schema_version": "datadiff-rq0-phenomenon-v1",
        "run_logs_scanned": len(run_files),
        "cases": cases,
        "cases_with_divergence": cases_with_divergence,
        "divergence_rate": rate,
        "findings": findings,
        "by_root_cause": dict(by_root.most_common()),
        "by_mismatch_class": dict(by_mismatch.most_common()),
        "by_oracle": dict(by_oracle.most_common()),
        "by_suspicious_backend": dict(by_backend.most_common()),
        "per_lane": {
            lane: {
                "cases": by_lane_cases[lane],
                "divergent": by_lane_divergent[lane],
                "rate": (by_lane_divergent[lane] / by_lane_cases[lane]) if by_lane_cases[lane] else 0.0,
            }
            for lane in sorted(by_lane_cases)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"run_logs={len(run_files)} cases={cases} divergent={cases_with_divergence} rate={rate:.4f}")
    print(f"top root causes: {by_root.most_common(6)}")
    print(f"top backends: {by_backend.most_common(4)}")
    print(f"json={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
