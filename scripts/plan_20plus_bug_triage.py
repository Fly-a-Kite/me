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
OWNED_DISCOVERY_CREDITS = {
    "datadiff_found",
    "datadiff_submitted",
    "user_found",
    "user_submitted",
}


def main() -> int:
    return run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> int:
    final_path = Path(str(args.final_readiness))
    final = load_json(final_path) if final_path.is_file() else {}
    bug_status = load_json(Path(str(args.bug_status))) if args.bug_status else {}
    confirmations = load_json(Path(str(args.latest_confirmations)))
    candidate_outcome_path = resolve_project_path(Path(str(args.candidate_outcomes)))
    candidate_outcomes = load_candidate_outcomes(candidate_outcome_path)
    rows = rank_candidates(
        final,
        bug_status=bug_status,
        confirmations=confirmations,
        candidate_outcomes=candidate_outcomes,
        limit=int(args.limit),
        candidate_source=str(args.candidate_source),
    )
    family_source = candidate_family_source(final, bug_status, candidate_source=str(args.candidate_source))[1]
    output_base = resolve_output_base(Path(str(args.output_base)))
    payload = {
        "schema_version": "bug-20plus-triage-plan-v1",
        "generated_at": utc_now_iso(),
        "goal": "Reach 20+ confirmed latest-version bug families by prioritizing high-yield candidate families.",
        "current_confirmed_count": len(confirmed_families(confirmations)),
        "target_confirmed_count": int(args.target_confirmed_count),
        "needed_confirmations": max(0, int(args.target_confirmed_count) - len(confirmed_families(confirmations))),
        "input_files": {
            "final_readiness": project_relative(final_path) if final_path.is_file() else "",
            "bug_status": project_relative(Path(str(args.bug_status))) if args.bug_status else "",
            "latest_confirmations": project_relative(Path(str(args.latest_confirmations))),
            "candidate_outcomes": project_relative(candidate_outcome_path) if candidate_outcome_path.is_file() else "",
        },
        "ranking_policy": {
            "candidate_family_source": family_source,
            "preferred_backends": list(PREFERRED_BACKENDS),
            "saturated_backends": sorted(SATURATED_BACKENDS),
            "noisy_roots": sorted(NOISY_ROOTS),
            "counting_policy": (
                "Do not count a row until the unique latest-version family was found/submitted by this project "
                "and upstream labels, acknowledges, or fixes it. Similar pre-existing upstream issues are dedup "
                "evidence only."
            ),
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
    candidate_outcomes: dict[str, dict[str, object]] | None = None,
    limit: int,
    candidate_source: str = "auto",
) -> list[dict[str, object]]:
    summary = final.get("summary", {}) if isinstance(final.get("summary"), dict) else {}
    rewardable, _source = candidate_family_source(final, bug_status, candidate_source=candidate_source)
    readiness_known = set((summary.get("known_saturated_live_candidate_families", {}) or {}).keys())
    confirmed = confirmed_families(confirmations)
    bug_summary = bug_status.get("summary", {}) if isinstance(bug_status.get("summary"), dict) else {}
    fresh = dict(bug_summary.get("recorded_fresh_candidate_families", {}) or {})
    evidence_sources = family_evidence_sources(bug_status)
    evidence_quality = {
        family: analyze_family_evidence(family, sources)
        for family, sources in evidence_sources.items()
    }
    outcomes = candidate_outcomes or {}
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
        quality = evidence_quality.get(family, {})
        risks.extend(dynamic_evidence_risks(quality))
        outcome = outcomes.get(family, {})
        risks.extend(dynamic_outcome_risks(outcome))
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
        score -= 28 * int("current_evidence_duplicate_column_schema_only" in risks)
        score -= 24 * int("current_evidence_nullable_join_key_boundary_only" in risks)
        score -= 28 * int("current_evidence_row_order_only_without_final_order_contract" in risks)
        score -= 18 * int("current_evidence_accept_reject_only" in risks)
        score -= 30 * int("current_evidence_marked_false_positive" in risks)
        score -= 18 * int("current_evidence_source_issue_only" in risks)
        score -= 30 * int("current_evidence_recheck_not_reproduced" in risks)
        score -= 45 * int("outcome_local_adapter_false_positive" in risks)
        score -= 42 * int("outcome_latest_recheck_no_longer_reproduces" in risks)
        score -= 38 * int("outcome_similar_existing_upstream_issue" in risks)
        score -= 34 * int("outcome_upstream_duplicate_or_not_planned" in risks)
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
                "evidence_sources": evidence_sources.get(family, [])[:5],
                "evidence_quality": quality,
                "candidate_outcome": outcome,
                "next_action": next_action(root, backends, risks),
            }
        )
    rows.sort(key=lambda row: (int(row["score"]), int(row["rewardable_live_count"]), str(row["family"])), reverse=True)
    return rows[: max(1, limit)]


def candidate_family_source(
    final: dict[str, Any],
    bug_status: dict[str, Any],
    *,
    candidate_source: str = "auto",
) -> tuple[dict[str, int], str]:
    summary = final.get("summary", {}) if isinstance(final.get("summary"), dict) else {}
    rewardable = _int_counter(summary.get("rewardable_live_candidate_families", {}) or {})
    if candidate_source == "final_readiness":
        return rewardable, "final_readiness.rewardable_live_candidate_families"
    bug_summary = bug_status.get("summary", {}) if isinstance(bug_status.get("summary"), dict) else {}
    recorded = _int_counter(bug_summary.get("recorded_fresh_candidate_families", {}) or {})
    fresh = _int_counter(bug_summary.get("fresh_candidate_families", {}) or {})
    if candidate_source == "bug_status":
        return recorded or fresh, "bug_status.recorded_fresh_candidate_families"
    if rewardable:
        return rewardable, "final_readiness.rewardable_live_candidate_families"
    return recorded or fresh, "bug_status.recorded_fresh_candidate_families"


def family_evidence_sources(bug_status: dict[str, Any]) -> dict[str, list[dict[str, object]]]:
    sources: dict[str, list[dict[str, object]]] = {}
    for section in ("fresh_candidate_evidence", "discovery_run_manifests", "discovery_campaign_manifests"):
        rows = bug_status.get(section, [])
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            families = item.get("fresh_candidate_bug_families", {})
            if not isinstance(families, dict):
                continue
            for family, raw_count in families.items():
                family_key = str(family).strip()
                if not family_key:
                    continue
                try:
                    count = int(raw_count or 0)
                except (TypeError, ValueError):
                    count = 0
                if count <= 0:
                    continue
                source = {
                    "section": section,
                    "path": str(item.get("path", "") or ""),
                    "source_run_file": str(item.get("source_run_file", "") or ""),
                    "generated_at": str(item.get("generated_at", "") or ""),
                    "count": count,
                }
                sources.setdefault(family_key, []).append(source)
    for family_sources in sources.values():
        family_sources.sort(
            key=lambda item: (
                int(item.get("count", 0) or 0),
                str(item.get("generated_at", "")),
                str(item.get("path", "")),
            ),
            reverse=True,
        )
    return sources


def analyze_family_evidence(family: str, sources: list[dict[str, object]], *, max_paths: int = 8) -> dict[str, object]:
    stats: dict[str, object] = {
        "candidate_row_count": 0,
        "duplicate_schema_row_count": 0,
        "nullable_join_key_boundary_row_count": 0,
        "finding_count": 0,
        "mismatch_class_counts": {},
        "false_positive_count": 0,
        "source_issue_count": 0,
        "recheck_reproduced_count": 0,
        "recheck_non_reproduced_count": 0,
        "sample_paths": [],
    }
    seen_paths: set[str] = set()
    for source in sources[:max_paths]:
        path_text = str(source.get("path", "") if isinstance(source, dict) else "").strip()
        if not path_text or path_text in seen_paths:
            continue
        seen_paths.add(path_text)
        path = resolve_project_path(Path(path_text))
        if not path.is_file():
            continue
        payload = load_json(path)
        rows = payload.get("candidate_rows", []) if isinstance(payload.get("candidate_rows", []), list) else []
        matched_in_path = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            findings = [
                finding
                for finding in row.get("findings", []) or []
                if isinstance(finding, dict) and finding_family(finding) == family
            ]
            if not findings:
                continue
            matched_in_path += 1
            stats["candidate_row_count"] = int(stats["candidate_row_count"]) + 1
            if row_has_duplicate_schema(row):
                stats["duplicate_schema_row_count"] = int(stats["duplicate_schema_row_count"]) + 1
            if row_has_nullable_join_key_boundary(row):
                stats["nullable_join_key_boundary_row_count"] = (
                    int(stats["nullable_join_key_boundary_row_count"]) + 1
                )
            recheck = row.get("candidate_recheck", {}) if isinstance(row.get("candidate_recheck", {}), dict) else {}
            if recheck.get("reproduced") or recheck.get("reproduced_keys"):
                stats["recheck_reproduced_count"] = int(stats["recheck_reproduced_count"]) + 1
            if recheck.get("non_reproduced_keys"):
                stats["recheck_non_reproduced_count"] = int(stats["recheck_non_reproduced_count"]) + 1
            for finding in findings:
                stats["finding_count"] = int(stats["finding_count"]) + 1
                mismatch = str(finding.get("mismatch_class", "") or "unknown")
                mismatch_counts = dict(stats["mismatch_class_counts"])
                mismatch_counts[mismatch] = int(mismatch_counts.get(mismatch, 0) or 0) + 1
                stats["mismatch_class_counts"] = mismatch_counts
                if bool(finding.get("false_positive")) or str(finding.get("false_positive_reason", "")).strip():
                    stats["false_positive_count"] = int(stats["false_positive_count"]) + 1
                if str(finding.get("source_issue", "")).strip():
                    stats["source_issue_count"] = int(stats["source_issue_count"]) + 1
        if matched_in_path:
            sample_paths = list(stats["sample_paths"])
            sample_paths.append(project_relative(path))
            stats["sample_paths"] = sample_paths[:5]
    return stats


def dynamic_evidence_risks(quality: dict[str, object]) -> list[str]:
    rows = int(quality.get("candidate_row_count", 0) or 0)
    findings = int(quality.get("finding_count", 0) or 0)
    if rows <= 0 or findings <= 0:
        return []
    risks: list[str] = []
    if int(quality.get("duplicate_schema_row_count", 0) or 0) == rows:
        risks.append("current_evidence_duplicate_column_schema_only")
    if int(quality.get("nullable_join_key_boundary_row_count", 0) or 0) == rows:
        risks.append("current_evidence_nullable_join_key_boundary_only")
    mismatch_counts = quality.get("mismatch_class_counts", {})
    if isinstance(mismatch_counts, dict):
        semantic_mismatches = sum(
            int(count or 0)
            for mismatch, count in mismatch_counts.items()
            if str(mismatch) in {"row_order", "value", "row_count"}
        )
        if semantic_mismatches > 0 and int(mismatch_counts.get("row_order", 0) or 0) == semantic_mismatches:
            risks.append("current_evidence_row_order_only_without_final_order_contract")
        if int(mismatch_counts.get("accept_reject", 0) or 0) == findings:
            risks.append("current_evidence_accept_reject_only")
    if int(quality.get("false_positive_count", 0) or 0) == findings:
        risks.append("current_evidence_marked_false_positive")
    if int(quality.get("source_issue_count", 0) or 0) == findings:
        risks.append("current_evidence_source_issue_only")
    recheck_non_reproduced = int(quality.get("recheck_non_reproduced_count", 0) or 0)
    recheck_reproduced = int(quality.get("recheck_reproduced_count", 0) or 0)
    if recheck_non_reproduced > 0 and recheck_reproduced == 0:
        risks.append("current_evidence_recheck_not_reproduced")
    return risks


def load_candidate_outcomes(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        return {}
    payload = load_json(path)
    rows = payload.get("outcomes", [])
    if not isinstance(rows, list):
        raise SystemExit(f"expected candidate outcome ledger with an outcomes array: {path}")
    outcomes: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("active") is False:
            continue
        family = str(row.get("family", "")).strip()
        outcome = str(row.get("outcome", "")).strip()
        if not family or not outcome:
            continue
        outcomes[family] = row
    return outcomes


def dynamic_outcome_risks(outcome: dict[str, object]) -> list[str]:
    name = str(outcome.get("outcome", "")).strip()
    if not name:
        return []
    risk_by_outcome = {
        "local_adapter_false_positive": "outcome_local_adapter_false_positive",
        "latest_recheck_no_longer_reproduces": "outcome_latest_recheck_no_longer_reproduces",
        "similar_existing_upstream_issue": "outcome_similar_existing_upstream_issue",
        "upstream_duplicate_or_not_planned": "outcome_upstream_duplicate_or_not_planned",
    }
    risk = risk_by_outcome.get(name)
    return [risk] if risk else [f"outcome_{name}"]


def finding_family(finding: dict[str, Any]) -> str:
    root = str(finding.get("root_cause", "")).strip()
    suspicious = finding.get("suspicious_backends", [])
    if not root or not isinstance(suspicious, list):
        return ""
    backends = ",".join(str(backend).strip() for backend in suspicious if str(backend).strip())
    return f"{root}@{backends}" if backends else ""


def row_has_duplicate_schema(row: dict[str, Any]) -> bool:
    case = row.get("case", {}) if isinstance(row.get("case", {}), dict) else {}
    tables = case.get("tables", []) if isinstance(case.get("tables", []), list) else []
    for table in tables:
        if not isinstance(table, dict):
            continue
        columns = table.get("columns", []) if isinstance(table.get("columns", []), list) else []
        names = [str(column.get("name", "")) for column in columns if isinstance(column, dict)]
        if len(names) != len(set(names)):
            return True
    return False


def row_has_nullable_join_key_boundary(row: dict[str, Any]) -> bool:
    case = row.get("case", {}) if isinstance(row.get("case", {}), dict) else {}
    tables = case.get("tables", []) if isinstance(case.get("tables", []), list) else []
    nullable_by_table: dict[str, dict[str, bool]] = {}
    for table in tables:
        if not isinstance(table, dict):
            continue
        table_name = str(table.get("name", "") or "")
        columns = table.get("columns", []) if isinstance(table.get("columns", []), list) else []
        nullable_by_table[table_name] = {
            str(column.get("name", "") or ""): bool(column.get("nullable"))
            for column in columns
            if isinstance(column, dict)
        }
    program = case.get("program", {}) if isinstance(case.get("program", {}), dict) else {}
    operations = program.get("operations", []) if isinstance(program.get("operations", []), list) else []
    for op in operations:
        if not isinstance(op, dict) or op.get("op") != "join":
            continue
        right_table = str(op.get("table", "") or "")
        left_keys = _join_key_list(op.get("left_on"))
        right_keys = _join_key_list(op.get("right_on"))
        for left_key, right_key in zip(left_keys, right_keys):
            if any(table_nullable.get(left_key, False) for table_nullable in nullable_by_table.values()):
                return True
            if nullable_by_table.get(right_table, {}).get(right_key, False):
                return True
    return False


def _join_key_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _int_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw_count in value.items():
        family = str(key).strip()
        if not family:
            continue
        try:
            count = int(raw_count or 0)
        except (TypeError, ValueError):
            count = 0
        if count > 0:
            out[family] = count
    return out


def split_family(family: str) -> tuple[str, list[str]]:
    root, sep, backend_text = family.partition("@")
    if not sep:
        return family, []
    return root, [item.strip() for item in backend_text.split(",") if item.strip()]


def priority_label(score: int, risks: list[str]) -> str:
    if "already_confirmed" in risks:
        return "exclude_confirmed"
    if "outcome_local_adapter_false_positive" in risks:
        return "deprioritize"
    if "outcome_latest_recheck_no_longer_reproduces" in risks:
        return "deprioritize"
    if "outcome_similar_existing_upstream_issue" in risks:
        return "deprioritize"
    if "outcome_upstream_duplicate_or_not_planned" in risks:
        return "deprioritize"
    if "current_evidence_duplicate_column_schema_only" in risks:
        return "deprioritize"
    if "current_evidence_nullable_join_key_boundary_only" in risks:
        return "deprioritize"
    if "current_evidence_row_order_only_without_final_order_contract" in risks:
        return "deprioritize"
    if "current_evidence_marked_false_positive" in risks:
        return "deprioritize"
    if "current_evidence_recheck_not_reproduced" in risks:
        return "deprioritize"
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
    if "outcome_local_adapter_false_positive" in risks:
        return "Do not submit; structured latest-version triage outcome marks this as a local adapter false positive."
    if "outcome_latest_recheck_no_longer_reproduces" in risks:
        return "Do not submit from stale artifacts; structured latest-version triage outcome says the current recheck no longer reproduces."
    if "outcome_similar_existing_upstream_issue" in risks:
        return "Do not open a duplicate; use as dedup/similarity evidence unless maintainers request a separate issue."
    if "outcome_upstream_duplicate_or_not_planned" in risks:
        return "Do not count toward the 20+ target; upstream duplicate/not-planned closure is not a project-owned confirmed family."
    if "current_evidence_duplicate_column_schema_only" in risks:
        return "Regenerate or repair this family with unique-column schemas before native DuckDB issue work."
    if "current_evidence_nullable_join_key_boundary_only" in risks:
        return "Minimize away nullable join-key boundaries or prove the backend behavior violates its documented null-join contract before submission."
    if "current_evidence_row_order_only_without_final_order_contract" in risks:
        return "Regenerate with a final ordering contract or value/row-count mismatch before upstream issue work."
    if "current_evidence_marked_false_positive" in risks:
        return "Do not submit; current evidence rows are marked false positive."
    if "current_evidence_recheck_not_reproduced" in risks:
        return "Do not submit from stale artifacts; current candidate recheck evidence no longer reproduces."
    if "current_evidence_source_issue_only" in risks:
        return "Treat as source-issue replay/dedup evidence; require a fresh independent latest reproducer before submission."
    if "current_evidence_accept_reject_only" in risks and len(backends) > 1:
        return "Minimize accept/reject evidence to one suspicious backend and prove the rejected query is valid before submission."
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
        if (
            isinstance(item, dict)
            and str(item.get("family", ""))
            and str(item.get("discovery_credit", "")).strip() in OWNED_DISCOVERY_CREDITS
        )
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


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def render_markdown(payload: dict[str, object]) -> str:
    rows = payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []
    lines = [
        "# 20+ Confirmed Bug Triage Plan",
        "",
        f"- Generated at: `{payload.get('generated_at', '')}`",
        f"- Current confirmed count: `{payload.get('current_confirmed_count', 0)}`",
        f"- Target confirmed count: `{payload.get('target_confirmed_count', 0)}`",
        f"- Needed confirmations: `{payload.get('needed_confirmations', 0)}`",
        f"- Candidate outcome ledger: `{candidate_outcome_ledger_text(payload)}`",
        "",
        "| priority | score | family | live count | fresh count | risks | source hints | next action |",
        "| --- | ---: | --- | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {priority} | {score} | `{family}` | {live} | {fresh} | {risks} | {sources} | {next_action} |".format(
                priority=row.get("priority", ""),
                score=row.get("score", ""),
                family=row.get("family", ""),
                live=row.get("rewardable_live_count", 0),
                fresh=row.get("bug_status_fresh_count", 0),
                risks=", ".join(row.get("risks", []) or []) or "none",
                sources=source_hint_text(row.get("evidence_sources", [])),
                next_action=row.get("next_action", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def candidate_outcome_ledger_text(payload: dict[str, object]) -> str:
    input_files = payload.get("input_files", {})
    if not isinstance(input_files, dict):
        return ""
    return str(input_files.get("candidate_outcomes", "") or "")


def source_hint_text(sources: object) -> str:
    if not isinstance(sources, list) or not sources:
        return ""
    hints = []
    for source in sources[:3]:
        if not isinstance(source, dict):
            continue
        path = str(source.get("path", "") or "")
        count = source.get("count", "")
        if path:
            hints.append(f"{path} ({count})")
    return "<br>".join(hints)


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
    parser.add_argument("--candidate-outcomes", default="reports/candidate-family-outcomes-current.json")
    parser.add_argument(
        "--candidate-source",
        choices=("auto", "final_readiness", "bug_status"),
        default="auto",
        help=(
            "Candidate-family source. auto uses final-readiness rewardable families when present "
            "and falls back to bug-status recorded fresh families."
        ),
    )
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
