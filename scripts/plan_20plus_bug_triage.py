#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
NOISY_ROOTS = {
    "conditional_expression",
    "nan_inf_semantics",
    "running_sum_precision",
}
PREFERRED_BACKENDS = ("duckdb", "pyarrow", "polars", "pandas")
SATURATED_BACKENDS = {"datafusion"}


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    final = load_json(Path(str(args.final_readiness)))
    bug_status = load_json(Path(str(args.bug_status))) if args.bug_status else {}
    confirmations = load_json(Path(str(args.latest_confirmations)))
    rows = rank_candidates(
        final,
        bug_status=bug_status,
        confirmations=confirmations,
        limit=int(args.limit),
    )
    output_base = resolve_output_base(Path(str(args.output_base)))
    payload = {
        "schema_version": "bug-20plus-triage-plan-v1",
        "generated_at": utc_now_iso(),
        "goal": "Reach 20+ confirmed latest-version bug families by prioritizing high-yield candidate families.",
        "current_confirmed_count": len(confirmed_families(confirmations)),
        "target_confirmed_count": int(args.target_confirmed_count),
        "needed_confirmations": max(0, int(args.target_confirmed_count) - len(confirmed_families(confirmations))),
        "input_files": {
            "final_readiness": project_relative(Path(str(args.final_readiness))),
            "bug_status": project_relative(Path(str(args.bug_status))) if args.bug_status else "",
            "latest_confirmations": project_relative(Path(str(args.latest_confirmations))),
        },
        "ranking_policy": {
            "preferred_backends": list(PREFERRED_BACKENDS),
            "saturated_backends": sorted(SATURATED_BACKENDS),
            "noisy_roots": sorted(NOISY_ROOTS),
            "counting_policy": "Do not count a row until upstream labels, acknowledges, or fixes the unique latest-version family.",
        },
        "candidates": rows,
    }
    write_outputs(output_base, payload)
    print(f"20+ triage plan json: {output_base.with_suffix('.json')}")
    print(f"20+ triage plan md:   {output_base.with_suffix('.md')}")
    return 0


def rank_candidates(
    final: dict[str, Any],
    *,
    bug_status: dict[str, Any],
    confirmations: dict[str, Any],
    limit: int,
) -> list[dict[str, object]]:
    summary = final.get("summary", {}) if isinstance(final.get("summary"), dict) else {}
    rewardable = dict(summary.get("rewardable_live_candidate_families", {}) or {})
    readiness_known = set((summary.get("known_saturated_live_candidate_families", {}) or {}).keys())
    confirmed = confirmed_families(confirmations)
    bug_summary = bug_status.get("summary", {}) if isinstance(bug_status.get("summary"), dict) else {}
    fresh = dict(bug_summary.get("recorded_fresh_candidate_families", {}) or {})
    bug_known = set()
    for key in ("known_saturated_families", "known_saturated_candidate_families"):
        value = bug_summary.get(key, {})
        if isinstance(value, dict):
            bug_known.update(value)
    rows: list[dict[str, object]] = []
    for family, count_value in rewardable.items():
        count = int(count_value or 0)
        root, backends = split_family(family)
        backend_set = set(backends)
        reasons: list[str] = []
        risks: list[str] = []
        if family in confirmed:
            risks.append("already_confirmed")
        if family in readiness_known or family in bug_known:
            risks.append("known_saturated_or_replay")
        if root in NOISY_ROOTS:
            risks.append("noisy_semantics_bucket")
        if backend_set and backend_set <= SATURATED_BACKENDS:
            risks.append("datafusion_already_well_covered")
        score = 0
        if any(backend in backend_set for backend in PREFERRED_BACKENDS):
            score += 35
            reasons.append("preferred_backend_diversity")
        if backend_set and not backend_set <= SATURATED_BACKENDS:
            score += 20
            reasons.append("non_datafusion_expansion")
        if len(backend_set) == 1:
            score += 10
            reasons.append("single_suspicious_backend")
        elif len(backend_set) > 1:
            score -= 6
            risks.append("multi_backend_expected_semantics_may_need_manual_spec_argument")
        if count >= 100:
            score += 12
            reasons.append("high_live_count")
        elif count >= 20:
            score += 8
            reasons.append("medium_live_count")
        elif count >= 5:
            score += 4
            reasons.append("low_but_repeated_live_count")
        if family in fresh:
            score += 12
            reasons.append("recorded_fresh_candidate")
        if root in {"coalesce_null_semantics", "drop_nulls_null_filter", "semi_join_membership", "union_all_row_append"}:
            score += 8
            reasons.append("clear_semantic_contract")
        if root in {"groupby_aggregation", "grouped_topk_null_sort_key", "filter_predicate", "ordering_or_limit"}:
            score += 5
            reasons.append("matches_prior_confirmed_bug_shape")
        score -= 25 * int("already_confirmed" in risks)
        score -= 18 * int("known_saturated_or_replay" in risks)
        score -= 20 * int("noisy_semantics_bucket" in risks)
        score -= 8 * int("datafusion_already_well_covered" in risks)
        rows.append(
            {
                "family": family,
                "root_cause": root,
                "backends": backends,
                "score": score,
                "rewardable_live_count": count,
                "bug_status_fresh_count": int(fresh.get(family, 0) or 0),
                "priority": priority_label(score, risks),
                "reasons": reasons,
                "risks": risks,
                "next_action": next_action(root, backends, risks),
            }
        )
    rows.sort(key=lambda row: (int(row["score"]), int(row["rewardable_live_count"]), str(row["family"])), reverse=True)
    return rows[: max(1, limit)]


def split_family(family: str) -> tuple[str, list[str]]:
    root, sep, backend_text = family.partition("@")
    if not sep:
        return family, []
    return root, [item.strip() for item in backend_text.split(",") if item.strip()]


def priority_label(score: int, risks: list[str]) -> str:
    if "already_confirmed" in risks:
        return "exclude_confirmed"
    if "known_saturated_or_replay" in risks or "noisy_semantics_bucket" in risks:
        return "deprioritize"
    if score >= 70:
        return "P0"
    if score >= 55:
        return "P1"
    return "P2"


def next_action(root: str, backends: list[str], risks: list[str]) -> str:
    if "already_confirmed" in risks:
        return "Do not submit; already counted in latest confirmations."
    if "known_saturated_or_replay" in risks:
        return "Deduplicate against old_issue/new_issue before any submission."
    if "noisy_semantics_bucket" in risks:
        return "Require a spec argument and tolerance/expected-semantics audit before issue drafting."
    backend_text = ",".join(backends) or "unknown"
    if len(backends) == 1:
        return f"Extract one minimal native {backend_text} reproducer and compare against at least one reference backend."
    return f"Minimize to a single suspicious backend from {backend_text}; only then draft upstream issue."


def confirmed_families(confirmations: dict[str, Any]) -> set[str]:
    return {
        str(item.get("family", ""))
        for item in confirmations.get("confirmations", []) or []
        if isinstance(item, dict) and str(item.get("family", ""))
    }


def write_outputs(output_base: Path, payload: dict[str, object]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    output_base.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_base.with_suffix(".md").write_text(render_markdown(payload), encoding="utf-8")


def resolve_output_base(path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parent != Path("."):
        return PROJECT_ROOT / path
    return DEFAULT_OUTPUT_DIR / path


def render_markdown(payload: dict[str, object]) -> str:
    rows = payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []
    lines = [
        "# 20+ Confirmed Bug Triage Plan",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Current confirmed count: `{payload.get('current_confirmed_count', 0)}`",
        f"- Target confirmed count: `{payload.get('target_confirmed_count', 0)}`",
        f"- Needed confirmations: `{payload.get('needed_confirmations', 0)}`",
        "",
        "| priority | score | family | live count | fresh count | risks | next action |",
        "| --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | {score} | `{family}` | {live} | {fresh} | {risks} | {next_action} |".format(
                priority=row.get("priority", ""),
                score=row.get("score", ""),
                family=row.get("family", ""),
                live=row.get("rewardable_live_count", 0),
                fresh=row.get("bug_status_fresh_count", 0),
                risks=", ".join(row.get("risks", []) or []) or "none",
                next_action=row.get("next_action", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return data


def project_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank latest-version candidate bug families for the 20+ confirmed target.")
    parser.add_argument("--final-readiness", default="reports/final-readiness-20260614T103547.json")
    parser.add_argument("--bug-status", default="reports/bug-status-20260614T110654.json")
    parser.add_argument("--latest-confirmations", default="experiments/latest_confirmations.json")
    parser.add_argument("--target-confirmed-count", type=int, default=20)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--output-base", default=f"bug-20plus-triage-plan-{utc_timestamp()}")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    sys.exit(main())
