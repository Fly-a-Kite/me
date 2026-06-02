from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.experiment_metadata import (
    canonical_comparison_role,
    experiment_row_group_id,
    experiment_row_variant_id,
    experiment_row_variant_key,
    experiment_row_variant_label,
    manifest_experiment_meta,
    row_factor_map,
    row_string_list,
    row_string_value,
    resolved_run_semantics,
)
from datadiff.normalizer import NormalizedResult, normalized_results_from_mapping
from datadiff.oracle import evaluate_case
from datadiff.reward import (
    aggregate_feedback_summary,
    aggregate_feedback_summaries,
    candidate_bug_family_keys as reward_candidate_bug_family_keys,
    is_candidate_bug_finding as reward_is_candidate_bug_finding,
    is_issue_replay_finding as reward_is_issue_replay_finding,
    is_known_saturated_candidate_bug_finding as reward_is_known_saturated_candidate_bug_finding,
    is_rewardable_candidate_bug_finding as reward_is_rewardable_candidate_bug_finding,
)
from datadiff.targets import target_context
from datadiff.util import REPORTS_DIR, RUNS_DIR, ensure_dirs, jsonl_log_stem, load_json, read_jsonl, run_meta_path


def latest_run_log_path() -> Path:
    files = sorted(
        [*RUNS_DIR.glob("run-*.jsonl"), *RUNS_DIR.glob("run-*.jsonl.gz")],
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not files:
        raise FileNotFoundError(f"no run logs found in {RUNS_DIR}")
    return files[-1]


def _string_list(values) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _counter_summary_limited(counter: Counter[str], *, limit: int = 8) -> str:
    if not counter:
        return "none"
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{key}:{count}" for key, count in items[:limit])


def _discovery_bias_summary_item(item) -> str:
    if not isinstance(item, dict):
        return ""
    targets = ",".join(_string_list(item.get("targets", []))) or "any-target"
    prefixes = ",".join(_string_list(item.get("feature_prefixes", []))) or "any-feature"
    flags = []
    if bool(item.get("keep_in_pool", False)):
        flags.append("keep")
    if float(item.get("score_bonus", 0.0) or 0.0):
        flags.append(f"score={float(item.get('score_bonus', 0.0) or 0.0):g}")
    if float(item.get("novelty_bonus", 0.0) or 0.0):
        flags.append(f"novelty={float(item.get('novelty_bonus', 0.0) or 0.0):g}")
    if float(item.get("contribution_bonus", 0.0) or 0.0):
        flags.append(f"contribution={float(item.get('contribution_bonus', 0.0) or 0.0):g}")
    if float(item.get("candidate_pool_bonus", 0.0) or 0.0):
        flags.append(f"pool={float(item.get('candidate_pool_bonus', 0.0) or 0.0):g}")
    suffix = f"|{'+'.join(flags)}" if flags else ""
    return f"{targets}|{prefixes}{suffix}"


def _config_discovery_biases_summary(value) -> str:
    items = [_discovery_bias_summary_item(item) for item in (value or []) if _discovery_bias_summary_item(item)]
    return "; ".join(items) if items else "none"


def write_run_report(run_file: Path | None = None, csv_limit: int | None = None) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_file = run_file or latest_run_log_path()
    rows = read_jsonl(run_file)
    run_stem = jsonl_log_stem(run_file)
    md_path = REPORTS_DIR / f"report-{run_stem}.md"
    csv_path = REPORTS_DIR / f"findings-{run_stem}.csv"

    status_counts = Counter(r.get("status", "unknown") for r in rows)
    finding_kinds = Counter()
    root_causes = Counter()
    oracle_counts = Counter()
    confidence_counts = Counter()
    triage_verdicts = Counter()
    discovery_origins = Counter()
    candidate_bug_families = Counter()
    false_positive_reasons = Counter()
    backend_status = Counter()
    quality_oracle_verdicts = Counter()
    quality_oracle_pass_fail = Counter()
    signatures = Counter(r.get("behavior_signature", "") for r in rows)
    examples: dict[str, list[dict]] = defaultdict(list)
    meta_path = run_meta_path(run_file)
    meta = load_json(meta_path) if meta_path.exists() else {}
    meta_config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    known_bug_families = list(meta_config.get("known_saturated_bug_families", []) or [])
    target_specs = meta.get("targets") or (rows[0].get("targets", []) if rows else [])
    target_ctx = meta.get("target_context", {}) if isinstance(meta.get("target_context", {}), dict) else {}
    target_families = Counter(target_ctx.get("families", [])) or Counter(
        target.get("family", "unknown") for target in target_specs
    )
    target_layers = Counter(target_ctx.get("layers", [])) or Counter(
        target.get("layer", "unknown") for target in target_specs
    )
    common_target_capabilities = (
        list(target_ctx.get("common_capabilities", []))
        or meta.get("common_capabilities")
        or _common_capabilities_from_specs(target_specs)
    )
    raw_new_behavior_cases = int(meta.get("new_behavior_cases", sum(1 for row in rows if row.get("is_new_behavior"))))
    signal_new_behavior_cases = int(
        meta.get(
            "signal_new_behavior_cases",
            sum(1 for row in rows if row.get("signal_new_behavior", row.get("is_new_behavior"))),
        )
    )

    for row in rows:
        if row.get("normalized"):
            for backend, norm in row.get("normalized", {}).items():
                backend_status[f"{backend}:{norm.get('status')}"] += 1
        else:
            for backend, status in row.get("backend_status", {}).items():
                backend_status[f"{backend}:{status}"] += 1
        for oracle in row.get("quality_oracles", []):
            quality_oracle_verdicts[f"{oracle.get('name', 'unknown')}:{oracle.get('verdict', 'unknown')}"] += 1
            outcome = "passed" if oracle.get("passed") else "failed"
            quality_oracle_pass_fail[f"{oracle.get('name', 'unknown')}:{outcome}"] += 1
        candidate_bug_families.update(_candidate_bug_family_keys(row.get("findings", []), known_bug_families))
        for finding in row.get("findings", []):
            finding_kinds[finding["kind"]] += 1
            root_causes[finding.get("root_cause", "unknown")] += 1
            oracle_counts[finding.get("oracle", "unknown")] += 1
            confidence_counts[finding.get("confidence", "unknown")] += 1
            triage_verdicts[finding.get("triage_verdict", "unclassified")] += 1
            discovery_origins[finding.get("discovery_origin", "legacy") or "legacy"] += 1
            if finding.get("false_positive_reason"):
                false_positive_reasons[finding.get("false_positive_reason", "")] += 1
            if len(examples[finding["kind"]]) < 3:
                examples[finding["kind"]].append(
                    {
                        "case_id": row["case"]["case_id"],
                        "seed": row["case"]["seed"],
                        "evidence": finding["evidence"],
                        "root_cause": finding.get("root_cause", "unknown"),
                        "oracle": finding.get("oracle", "unknown"),
                        "confidence": finding.get("confidence", "unknown"),
                        "triage_verdict": finding.get("triage_verdict", "unclassified"),
                        "bug_dir": row.get("bug_dir", ""),
                    }
                )

    total = len(rows)
    bug_rows = [r for r in rows if r.get("findings")]
    lines = [
        "# DataDiffFuzz Report",
        "",
        f"- Run log: `{run_file}`",
        f"- Cases: {total}",
        f"- Bug-triggering cases: {len(bug_rows)}",
        f"- Bug rate: {len(bug_rows) / total:.1%}" if total else "- Bug rate: n/a",
        f"- Unique behavior signatures: {len(signatures)}",
        f"- Raw new behavior cases: {raw_new_behavior_cases}",
        f"- Signal new behavior cases: {signal_new_behavior_cases}",
        f"- Elapsed seconds: {meta.get('elapsed_s', 'n/a')}",
        f"- Throughput cases/s: {meta.get('throughput_cases_s', 'n/a')}",
        f"- Backends: {', '.join(meta.get('backends', [])) if meta.get('backends') else 'n/a'}",
        f"- Target families: {_counter_summary(target_families)}",
        f"- Target layers: {_counter_summary(target_layers)}",
        f"- Common target capabilities: {len(common_target_capabilities)}",
        f"- Findings CSV limit: {'none' if csv_limit is None else max(0, csv_limit)}",
        "",
        "## Status",
    ]
    for key, count in status_counts.most_common():
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Backend Status")
    for key, count in backend_status.most_common():
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## Finding Kinds")
    if finding_kinds:
        for key, count in finding_kinds.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Root Causes")
    if root_causes:
        for key, count in root_causes.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Oracle Sources")
    if oracle_counts:
        for key, count in oracle_counts.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Confidence")
    if confidence_counts:
        for key, count in confidence_counts.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Triage Verdicts")
    if triage_verdicts:
        for key, count in triage_verdicts.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Discovery Origins")
    if discovery_origins:
        for key, count in discovery_origins.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Candidate Bug Families")
    if candidate_bug_families:
        for key, count in candidate_bug_families.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## False Positive Reasons")
    if false_positive_reasons:
        for key, count in false_positive_reasons.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Target Capability Intersection")
    if common_target_capabilities:
        for capability in common_target_capabilities:
            lines.append(f"- {capability}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Quality Oracles")
    if quality_oracle_verdicts:
        lines.append("Verdicts:")
        for key, count in quality_oracle_verdicts.most_common():
            lines.append(f"- {key}: {count}")
        lines.append("")
        lines.append("Pass/fail:")
        for key, count in quality_oracle_pass_fail.most_common():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Representative Evidence")
    if not examples:
        lines.append("- No findings.")
    for kind, items in examples.items():
        lines.append(f"### {kind}")
        for item in items:
            lines.append(
                f"- `{item['case_id']}` seed={item['seed']} "
                f"root={item['root_cause']} oracle={item['oracle']} confidence={item['confidence']} "
                f"triage={item['triage_verdict']}: "
                f"{item['evidence']} "
                f"artifact=`{item['bug_dir']}`"
            )
        lines.append("")
    if not examples:
        lines.append("")
    lines.append("## Ablation-Oriented Interpretation")
    lines.append("")
    lines.append(
        "Each row records the active modules in `config`: type-aware generation, "
        "semantic normalization, differential oracle, feedback, reducer, and artifact generation. "
        "Run the same seed range with modules disabled to quantify their contribution."
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "seed",
                "kind",
                "severity",
                "root_cause",
                "oracle",
                "confidence",
                "triage_verdict",
                "paper_status",
                "triage_confidence",
                "false_positive",
                "false_positive_reason",
                "suspicious_backends",
                "signature",
                "evidence",
                "triage_evidence",
                "bug_dir",
            ],
        )
        writer.writeheader()
        written = 0
        max_rows = None if csv_limit is None else max(0, int(csv_limit))
        for row in rows:
            for finding in row.get("findings", []):
                if max_rows is not None and written >= max_rows:
                    break
                writer.writerow(
                    {
                        "case_id": row["case"]["case_id"],
                        "seed": row["case"]["seed"],
                        "kind": finding["kind"],
                        "severity": finding["severity"],
                        "root_cause": finding.get("root_cause", "unknown"),
                        "oracle": finding.get("oracle", "unknown"),
                        "confidence": finding.get("confidence", "unknown"),
                        "triage_verdict": finding.get("triage_verdict", "unclassified"),
                        "paper_status": finding.get("paper_status", "unclassified"),
                        "triage_confidence": finding.get("triage_confidence", "low"),
                        "false_positive": finding.get("false_positive", False),
                        "false_positive_reason": finding.get("false_positive_reason", ""),
                        "suspicious_backends": ",".join(finding.get("suspicious_backends", [])),
                        "signature": finding["signature"],
                        "evidence": finding["evidence"],
                        "triage_evidence": finding.get("triage_evidence", ""),
                        "bug_dir": row.get("bug_dir", ""),
                    }
                )
                written += 1
            if max_rows is not None and written >= max_rows:
                break
    return md_path, csv_path


def latest_experiment_manifest_path() -> Path:
    files = sorted(RUNS_DIR.glob("experiment-*.json"))
    if not files:
        raise FileNotFoundError(f"no experiment manifests found in {RUNS_DIR}")
    return files[-1]


def write_experiment_summary_report(
    manifest_file: Path | None = None,
    *,
    refresh: bool = False,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest_path()
    manifest = load_json(manifest_file)
    md_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}.md"
    csv_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}.csv"
    aggregate_csv_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}-aggregates.csv"
    aggregate_json_path = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}-aggregates.json"
    experiment_meta = manifest_experiment_meta(manifest)

    rows = []
    adaptive_state_by_arm = _adaptive_state_by_arm(manifest.get("adaptive_state", []))
    for run in manifest.get("runs", []):
        run_file = Path(run["run_file"])
        run_rows = read_jsonl(run_file)
        if refresh:
            run_rows = [_refresh_summary_row_findings(row) for row in run_rows]
        meta_path = run_meta_path(run_file)
        meta = load_json(meta_path) if meta_path.exists() else {}
        meta_config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
        known_bug_families = list(meta_config.get("known_saturated_bug_families", []) or [])
        replay_filter = (
            meta.get("replay_bug_filter", {}) if isinstance(meta.get("replay_bug_filter", {}), dict) else {}
        )
        family_saturation_filter = (
            meta.get("family_saturation_filter", {})
            if isinstance(meta.get("family_saturation_filter", {}), dict)
            else {}
        )
        preflight = meta.get("preflight", {}) if isinstance(meta.get("preflight", {}), dict) else {}
        findings = [finding for row in run_rows for finding in row.get("findings", [])]
        unique_finding_signatures = {f.get("signature", "") for f in findings}
        root_causes = Counter(f.get("root_cause", "unknown") for f in findings)
        finding_kinds = Counter(f.get("kind", "unknown") for f in findings)
        triage_verdicts = Counter(f.get("triage_verdict", "unclassified") for f in findings)
        discovery_origins = Counter(f.get("discovery_origin", "legacy") or "legacy" for f in findings)
        issue_replay_candidate_count = sum(
            1 for finding in findings if _is_candidate_bug_finding(finding) and _is_issue_replay_finding(finding)
        )
        known_saturated_candidate_count = sum(
            1 for finding in findings if _is_known_saturated_candidate_bug_finding(finding, known_bug_families)
        )
        rewardable_candidate_count = sum(
            1 for finding in findings if _is_rewardable_candidate_bug_finding(finding, known_bug_families)
        )
        candidate_bug_families = Counter()
        for row in run_rows:
            candidate_bug_families.update(_candidate_bug_family_keys(row.get("findings", []), known_bug_families))
        run_target_ctx = meta.get("target_context", {}) if isinstance(meta.get("target_context", {}), dict) else {}
        target_families = Counter(run_target_ctx.get("families", [])) or Counter(
            target.get("family", "unknown") for target in meta.get("targets", [])
        )
        stage_profile = meta.get("stage_profile", {}) if isinstance(meta.get("stage_profile", {}), dict) else {}
        stage_totals = stage_profile.get("totals_ms", {}) if isinstance(stage_profile.get("totals_ms", {}), dict) else {}
        stage_avg = stage_profile.get("avg_ms_per_case", {}) if isinstance(stage_profile.get("avg_ms_per_case", {}), dict) else {}
        stage_share = stage_profile.get("share_of_total", {}) if isinstance(stage_profile.get("share_of_total", {}), dict) else {}
        total = len(run_rows)
        bug_cases = sum(1 for row in run_rows if row.get("findings"))
        run_log_bytes = _file_size_bytes(run_file)
        artifact_bytes = _artifact_bytes_from_rows(run_rows)
        evidence_bytes = run_log_bytes + artifact_bytes
        candidate_bug_cases = sum(
            1
            for row in run_rows
            if any(
                _is_candidate_bug_finding(finding)
                for finding in row.get("findings", [])
            )
        )
        first_finding_case_index = _first_case_index(run_rows, lambda finding: True)
        first_candidate_bug_case_index = _first_case_index(
            run_rows,
            _is_candidate_bug_finding,
        )
        first_candidate_bug_elapsed_s = _first_case_elapsed_s(run_rows, _is_candidate_bug_finding)
        candidate_bug_discovery_auc = _candidate_bug_discovery_auc(run_rows)
        guidance_metrics = _guidance_metrics(run_rows)
        feedback_metrics = _feedback_metrics(run_rows, known_bug_families)
        new_behavior_cases = int(meta.get("new_behavior_cases", sum(1 for row in run_rows if row.get("is_new_behavior"))))
        signal_new_behavior_cases = int(
            meta.get(
                "signal_new_behavior_cases",
                sum(1 for row in run_rows if row.get("signal_new_behavior", row.get("is_new_behavior"))),
            )
        )
        arm_state = adaptive_state_by_arm.get(str(run.get("schedule_arm_id", "")), {})
        run_semantics = resolved_run_semantics(run, experiment_meta)
        rows.append(
            {
                "target_suite": run.get("target_suite", manifest.get("target_suite", "")),
                "preset": run.get("preset", ""),
                "matrix_id": run_semantics["matrix_id"],
                "matrix_title": run_semantics["matrix_title"],
                "comparison_group": run_semantics["comparison_group"],
                "variant_id": run_semantics["variant_id"],
                "variant_title": run_semantics["variant_title"],
                "variant_label": experiment_row_variant_label(run_semantics),
                "variant_group_id": "|".join(experiment_row_group_id(run_semantics)),
                "base_preset": run_semantics["base_preset"],
                "comparison_role": run_semantics["comparison_role"],
                "canonical_comparison_role": run_semantics["canonical_comparison_role"],
                "component_focus": run_semantics["component_focus"],
                "overlays": ",".join(run_semantics["overlays"]),
                "semantic_focus_families": ",".join(run_semantics["semantic_focus_families"]),
                "semantic_focus_signals": ",".join(run_semantics["semantic_focus_signals"]),
                "factors": json.dumps(run_semantics["factors"], ensure_ascii=False, sort_keys=True),
                "scope_kind": run_semantics["scope_kind"],
                "oracle_profile": run_semantics["oracle_profile"],
                "rq_tags": ",".join(run_semantics["rq_tags"]),
                "analysis_tags": ",".join(run_semantics["analysis_tags"]),
                "counts_as_real_bugs": run_semantics["counts_as_real_bugs"],
                "seed": run.get("seed", ""),
                "evidence_mode": run.get("evidence_mode", manifest.get("evidence_mode", "")),
                "known_bug_id": run.get("known_bug_id", manifest.get("known_bug_id", "")),
                "target_version": run.get("target_version", manifest.get("target_version", "")),
                "batch_index": run.get("batch_index", ""),
                "schedule_arm_id": run.get("schedule_arm_id", ""),
                "scheduler_reward": run.get("scheduler_reward", ""),
                "scheduler_mean_reward": arm_state.get("mean_reward", ""),
                "scheduler_reward_signal": arm_state.get("reward_signal", ""),
                "scheduler_last_reward": arm_state.get("last_reward", ""),
                "scheduler_pulls": arm_state.get("pulls", ""),
                "scheduler_stale_batches": arm_state.get("stale_batches", ""),
                "enable_replay_bug": bool(meta_config.get("enable_replay_bug", False)),
                "replay_filter_enabled": replay_filter.get("enabled", ""),
                "replay_filter_filtered_candidates": int(replay_filter.get("filtered_candidates", 0) or 0),
                "replay_filter_fallback_candidates": int(replay_filter.get("fallback_candidates", 0) or 0),
                "family_saturation_filter_enabled": family_saturation_filter.get("enabled", ""),
                "family_saturation_filter_filtered_candidates": int(
                    family_saturation_filter.get("filtered_candidates", 0) or 0
                ),
                "family_saturation_filter_fallback_candidates": int(
                    family_saturation_filter.get("fallback_candidates", 0) or 0
                ),
                "configured_guidance_targets": ",".join(_string_list(meta_config.get("guidance_targets", []))),
                "configured_effective_guidance_targets": ",".join(
                    _string_list(meta_config.get("effective_guidance_targets", []))
                ),
                "configured_semantic_focus_families": ",".join(_string_list(meta_config.get("semantic_focus_families", []))),
                "configured_semantic_focus_signals": ",".join(_string_list(meta_config.get("semantic_focus_signals", []))),
                "configured_discovery_biases": _config_discovery_biases_summary(meta_config.get("discovery_biases", [])),
                "cases": total,
                "run_log_bytes": run_log_bytes,
                "run_log_bytes_per_case": run_log_bytes / total if total else 0.0,
                "artifact_bytes": artifact_bytes,
                "artifact_bytes_per_case": artifact_bytes / total if total else 0.0,
                "evidence_bytes": evidence_bytes,
                "evidence_bytes_per_case": evidence_bytes / total if total else 0.0,
                "bug_cases": bug_cases,
                "bug_rate": bug_cases / total if total else 0.0,
                "candidate_bug_cases": candidate_bug_cases,
                "candidate_bug_case_rate": candidate_bug_cases / total if total else 0.0,
                "first_finding_case_index": first_finding_case_index,
                "first_candidate_bug_case_index": first_candidate_bug_case_index,
                "first_candidate_bug_elapsed_s": first_candidate_bug_elapsed_s,
                "candidate_bug_discovery_auc": candidate_bug_discovery_auc,
                "findings": len(findings),
                "unique_findings": len(unique_finding_signatures),
                "new_behavior_cases": new_behavior_cases,
                "new_behavior_rate": new_behavior_cases / total if total else 0.0,
                "signal_new_behavior_cases": signal_new_behavior_cases,
                "signal_new_behavior_rate": signal_new_behavior_cases / total if total else 0.0,
                "preflight_repaired_cases": int(preflight.get("repaired_cases", 0) or 0),
                "preflight_fallback_cases": int(preflight.get("fallback_cases", 0) or 0),
                "preflight_invalid_cases": int(preflight.get("invalid_cases", 0) or 0),
                "preflight_repaired_rate": (int(preflight.get("repaired_cases", 0) or 0) / total if total else 0.0),
                "preflight_fallback_rate": (int(preflight.get("fallback_cases", 0) or 0) / total if total else 0.0),
                "preflight_invalid_rate": (int(preflight.get("invalid_cases", 0) or 0) / total if total else 0.0),
                "elapsed_s": meta.get("elapsed_s", ""),
                "throughput_cases_s": meta.get("throughput_cases_s", ""),
                "stage_generate_mutate_total_ms": float(stage_totals.get("generate_mutate_ms", 0.0) or 0.0),
                "stage_backend_execution_total_ms": float(stage_totals.get("backend_execution_ms", 0.0) or 0.0),
                "stage_normalize_total_ms": float(stage_totals.get("normalize_ms", 0.0) or 0.0),
                "stage_oracle_classification_total_ms": float(stage_totals.get("oracle_classification_ms", 0.0) or 0.0),
                "stage_scheduler_feedback_total_ms": float(stage_totals.get("scheduler_feedback_ms", 0.0) or 0.0),
                "stage_logging_artifact_total_ms": float(stage_totals.get("logging_artifact_ms", 0.0) or 0.0),
                "stage_total_case_wall_ms": float(stage_totals.get("total_case_wall_ms", 0.0) or 0.0),
                "stage_generate_mutate_avg_ms": float(stage_avg.get("generate_mutate_ms", 0.0) or 0.0),
                "stage_backend_execution_avg_ms": float(stage_avg.get("backend_execution_ms", 0.0) or 0.0),
                "stage_normalize_avg_ms": float(stage_avg.get("normalize_ms", 0.0) or 0.0),
                "stage_oracle_classification_avg_ms": float(stage_avg.get("oracle_classification_ms", 0.0) or 0.0),
                "stage_scheduler_feedback_avg_ms": float(stage_avg.get("scheduler_feedback_ms", 0.0) or 0.0),
                "stage_logging_artifact_avg_ms": float(stage_avg.get("logging_artifact_ms", 0.0) or 0.0),
                "stage_generate_mutate_share": float(stage_share.get("generate_mutate_ms", 0.0) or 0.0),
                "stage_backend_execution_share": float(stage_share.get("backend_execution_ms", 0.0) or 0.0),
                "stage_normalize_share": float(stage_share.get("normalize_ms", 0.0) or 0.0),
                "stage_oracle_classification_share": float(stage_share.get("oracle_classification_ms", 0.0) or 0.0),
                "stage_scheduler_feedback_share": float(stage_share.get("scheduler_feedback_ms", 0.0) or 0.0),
                "stage_logging_artifact_share": float(stage_share.get("logging_artifact_ms", 0.0) or 0.0),
                **guidance_metrics,
                **feedback_metrics,
                "top_root_causes": _counter_summary(root_causes),
                "top_finding_kinds": _counter_summary(finding_kinds),
                "top_triage_verdicts": _counter_summary(triage_verdicts),
                "top_discovery_origins": _counter_summary(discovery_origins),
                "candidate_bug_families": len(candidate_bug_families),
                "top_candidate_bug_families": _counter_summary(candidate_bug_families),
                "candidate_implementation_bug_count": triage_verdicts["candidate_implementation_bug"],
                "rewardable_candidate_implementation_bug_count": rewardable_candidate_count,
                "issue_replay_candidate_bug_count": issue_replay_candidate_count,
                "known_saturated_candidate_bug_count": known_saturated_candidate_count,
                "documented_semantic_divergence_count": triage_verdicts["documented_semantic_divergence"],
                "expected_semantic_divergence_count": triage_verdicts["expected_semantic_divergence"],
                "semantic_divergence_needs_confirmation_count": triage_verdicts[
                    "semantic_divergence_needs_confirmation"
                ],
                "generator_false_positive_count": triage_verdicts["generator_false_positive"],
                "normalizer_false_positive_count": triage_verdicts["normalizer_false_positive"],
                "needs_manual_confirmation_count": triage_verdicts["needs_manual_confirmation"],
                "target_families": _counter_summary(target_families),
                "backends": ",".join(run.get("backends", manifest.get("backends", []))),
                "run_file": str(run_file),
                "report": run.get("report", ""),
            }
        )

    replay_policy = manifest.get("replay_bug_policy", {})
    replay_source_issues = replay_policy.get("source_issues", []) if isinstance(replay_policy, dict) else []
    lines = [
        "# DataDiffFuzz Experiment Summary",
        "",
        f"- Manifest: `{manifest_file}`",
        f"- Runs: {len(rows)}",
        f"- Matrix id: {experiment_meta.get('matrix_id', '') or 'n/a'}",
        f"- Comparison group: {experiment_meta.get('comparison_group', '') or 'n/a'}",
        f"- Variants requested: {', '.join(manifest.get('presets', []))}",
        f"- Seeds: {', '.join(str(s) for s in manifest.get('seeds', []))}",
        f"- Backends: {', '.join(manifest.get('backends', []))}",
        f"- Target suite: {manifest.get('target_suite', 'n/a')}",
        f"- Target suites: {', '.join(manifest.get('target_suites', [])) or manifest.get('target_suite', 'n/a')}",
        f"- Target families: {_counter_summary(Counter(target.get('family', 'unknown') for target in manifest.get('targets', [])))}",
        f"- Common target capabilities: {len(manifest.get('common_capabilities', []))}",
        f"- Schedule: {manifest.get('schedule', 'matrix_order')}",
        f"- Evidence mode: {manifest.get('evidence_mode', 'live')}",
        (
            f"- Replay bug policy: enable_replay_bug={str(bool(replay_policy.get('enable_replay_bug', False))).lower()}, "
            f"source_issues={len(replay_source_issues) if isinstance(replay_source_issues, list) else 0}"
        ),
        f"- Known bug id: {manifest.get('known_bug_id', '') or 'n/a'}",
        f"- Target version: {manifest.get('target_version', '') or 'n/a'}",
        f"- Local source scheduler: {manifest.get('local_source_scheduler', {'enabled': False, 'exploration_weight': ''})}",
        f"- Aggregate CSV: `{aggregate_csv_path}`",
        f"- Aggregate JSON: `{aggregate_json_path}`",
        "- Triage columns distinguish candidate implementation bugs from documented/expected semantic divergences and oracle false positives.",
        f"- Refreshed with current oracle: {'yes' if refresh else 'no'}",
        "",
        "## Runs",
        "",
        "| target suite | variant | preset | seed | batch | arm | reward | reward signal | mean reward | pulls | stale | cases | findings | candidate bugs | rewardable candidates | issue replays | known families | candidate case % | first candidate | first candidate s | discovery AUC | semantic divs | false positives | raw new behavior % | signal new behavior % | cases/s | data sensitivity | path proxy | frontier | discovery bonus | pool bonus | stale penalty | contribution | pruned % | roots | triage | origins |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        semantic_divergence_count = (
            row["documented_semantic_divergence_count"]
            + row["expected_semantic_divergence_count"]
            + row["semantic_divergence_needs_confirmation_count"]
        )
        false_positive_count = row["generator_false_positive_count"] + row["normalizer_false_positive_count"]
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {seed} | {batch_index} | {schedule_arm_id} | {scheduler_reward} | {scheduler_reward_signal} | {scheduler_mean_reward} | {scheduler_pulls} | {scheduler_stale_batches} | {cases} | {findings} | "
            "{candidate_implementation_bug_count} | {rewardable_candidate_implementation_bug_count} | {issue_replay_candidate_bug_count} | {known_saturated_candidate_bug_count} | {candidate_bug_case_rate} | "
            "{first_candidate_bug_case_index} | {first_candidate_bug_elapsed_s} | {candidate_bug_discovery_auc} | {semantic_divergence_count} | "
            "{false_positive_count} | {new_behavior_rate} | {signal_new_behavior_rate} | {throughput_cases_s} | {avg_data_sensitivity} | "
            "{avg_path_coverage_proxy} | {avg_frontier_conformance} | "
            "{avg_discovery_diversity_bonus} | {avg_candidate_pool_diversity_bonus} | {avg_discovery_stale_penalty} | "
            "{avg_contribution_potential} | {pruned_candidate_rate} | {top_root_causes} | {top_triage_verdicts} | {top_discovery_origins} |".format(
                **{
                    **row,
                    "semantic_divergence_count": semantic_divergence_count,
                    "false_positive_count": false_positive_count,
                    "batch_index": _fmt_optional_int(row["batch_index"]),
                    "scheduler_reward": _fmt_optional_float(row["scheduler_reward"]),
                    "scheduler_reward_signal": _fmt_optional_float(row["scheduler_reward_signal"]),
                    "scheduler_mean_reward": _fmt_optional_float(row["scheduler_mean_reward"]),
                    "scheduler_pulls": _fmt_optional_int(row["scheduler_pulls"]),
                    "scheduler_stale_batches": _fmt_optional_int(row["scheduler_stale_batches"]),
                    "candidate_bug_case_rate": _fmt_percent(row["candidate_bug_case_rate"]),
                    "first_candidate_bug_case_index": _fmt_optional_int(row["first_candidate_bug_case_index"]),
                    "first_candidate_bug_elapsed_s": _fmt_optional_float(row["first_candidate_bug_elapsed_s"]),
                    "candidate_bug_discovery_auc": _fmt_float(row["candidate_bug_discovery_auc"]),
                    "new_behavior_rate": _fmt_percent(row["new_behavior_rate"]),
                    "signal_new_behavior_rate": _fmt_percent(row["signal_new_behavior_rate"]),
                    "throughput_cases_s": _fmt_float(row["throughput_cases_s"]),
                    "avg_data_sensitivity": _fmt_float(row["avg_data_sensitivity"]),
                    "avg_path_coverage_proxy": _fmt_float(row["avg_path_coverage_proxy"]),
                    "avg_frontier_conformance": _fmt_float(row["avg_frontier_conformance"]),
                    "avg_discovery_diversity_bonus": _fmt_float(row["avg_discovery_diversity_bonus"]),
                    "avg_candidate_pool_diversity_bonus": _fmt_float(row["avg_candidate_pool_diversity_bonus"]),
                    "avg_discovery_stale_penalty": _fmt_float(row["avg_discovery_stale_penalty"]),
                    "avg_contribution_potential": _fmt_float(row["avg_contribution_potential"]),
                    "pruned_candidate_rate": _fmt_percent(row["pruned_candidate_rate"]),
                }
            )
        )
    aggregate_rows = _aggregate_experiment_rows(rows)
    lines.extend(
        [
            "",
            "## Aggregates",
            "",
            "| target suite | variant | preset | runs | cases | findings | candidate bugs | rewardable candidates | issue replays | known families | candidate case % | candidate cases/s | median first candidate | median first s | avg discovery AUC | avg reward | avg reward signal | avg mean reward | semantic divs | false positives | avg raw new behavior % | avg signal new behavior % | avg cases/s | avg data sensitivity | avg path proxy | avg discovery bonus | avg pool bonus | avg stale penalty | avg discovery buckets |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {runs} | {cases} | {findings} | "
            "{candidate_implementation_bug_count} | {rewardable_candidate_implementation_bug_count} | {issue_replay_candidate_bug_count} | {known_saturated_candidate_bug_count} | {candidate_bug_case_rate} | "
            "{candidate_bug_cases_per_s} | {median_first_candidate_bug_case_index} | "
            "{median_first_candidate_bug_elapsed_s} | {avg_candidate_bug_discovery_auc} | {avg_scheduler_reward} | {avg_scheduler_reward_signal} | {avg_scheduler_mean_reward} | "
            "{semantic_divergence_count} | {false_positive_count} | {avg_new_behavior_rate} | {avg_signal_new_behavior_rate} | "
            "{avg_throughput_cases_s} | {avg_data_sensitivity} | {avg_path_coverage_proxy} | "
            "{avg_discovery_diversity_bonus} | {avg_candidate_pool_diversity_bonus} | "
            "{avg_discovery_stale_penalty} | {avg_discovery_bucket_count} |".format(
                **{
                    **row,
                    "candidate_bug_case_rate": _fmt_percent(row["candidate_bug_case_rate"]),
                    "candidate_bug_cases_per_s": _fmt_float(row["candidate_bug_cases_per_s"]),
                    "median_first_candidate_bug_case_index": _fmt_optional_number(
                        row["median_first_candidate_bug_case_index"]
                    ),
                    "median_first_candidate_bug_elapsed_s": _fmt_optional_number(
                        row["median_first_candidate_bug_elapsed_s"]
                    ),
                    "avg_candidate_bug_discovery_auc": _fmt_float(row["avg_candidate_bug_discovery_auc"]),
                    "avg_scheduler_reward": _fmt_optional_float(row["avg_scheduler_reward"]),
                    "avg_scheduler_reward_signal": _fmt_optional_float(row["avg_scheduler_reward_signal"]),
                    "avg_scheduler_mean_reward": _fmt_optional_float(row["avg_scheduler_mean_reward"]),
                    "avg_new_behavior_rate": _fmt_percent(row["avg_new_behavior_rate"]),
                    "avg_signal_new_behavior_rate": _fmt_percent(row["avg_signal_new_behavior_rate"]),
                    "avg_throughput_cases_s": _fmt_float(row["avg_throughput_cases_s"]),
                    "avg_data_sensitivity": _fmt_float(row["avg_data_sensitivity"]),
                    "avg_path_coverage_proxy": _fmt_float(row["avg_path_coverage_proxy"]),
                    "avg_discovery_diversity_bonus": _fmt_float(row["avg_discovery_diversity_bonus"]),
                    "avg_candidate_pool_diversity_bonus": _fmt_float(row["avg_candidate_pool_diversity_bonus"]),
                    "avg_discovery_stale_penalty": _fmt_float(row["avg_discovery_stale_penalty"]),
                    "avg_discovery_bucket_count": _fmt_float(row["avg_discovery_bucket_count"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Closed-Loop Feedback",
            "",
            "| target suite | variant | preset | runs | cases | raw novelty % | signal novelty % | feedback mutation % | affinity-hit % | avg selected-op score | corpus store % | quality pass % | productive mutation % | guided productive % | source adj/case | guidance adj/case | seed delta/case |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {runs} | {cases} | {avg_new_behavior_rate} | {avg_signal_new_behavior_rate} | {feedback_mutation_case_rate} | "
            "{feedback_operator_affinity_hit_rate} | {feedback_selected_operator_score_avg} | {feedback_corpus_store_rate} | {quality_pass_rate} | {productive_mutation_rate} | "
            "{guided_productive_rate} | {source_reward_adjustment_per_case} | "
            "{guidance_reward_adjustment_per_case} | {seed_schedule_delta_per_case} |".format(
                **{
                    **row,
                    "avg_new_behavior_rate": _fmt_percent(row["avg_new_behavior_rate"]),
                    "avg_signal_new_behavior_rate": _fmt_percent(row["avg_signal_new_behavior_rate"]),
                    "feedback_mutation_case_rate": _fmt_percent(row["feedback_mutation_case_rate"]),
                    "feedback_operator_affinity_hit_rate": _fmt_percent(row["feedback_operator_affinity_hit_rate"]),
                    "feedback_selected_operator_score_avg": _fmt_float(row["feedback_selected_operator_score_avg"]),
                    "feedback_corpus_store_rate": _fmt_percent(row["feedback_corpus_store_rate"]),
                    "quality_pass_rate": _fmt_percent(row["quality_pass_rate"]),
                    "productive_mutation_rate": _fmt_percent(row["productive_mutation_rate"]),
                    "guided_productive_rate": _fmt_percent(row["guided_productive_rate"]),
                    "source_reward_adjustment_per_case": _fmt_float(row["source_reward_adjustment_per_case"]),
                    "guidance_reward_adjustment_per_case": _fmt_float(
                        row["guidance_reward_adjustment_per_case"]
                    ),
                    "seed_schedule_delta_per_case": _fmt_float(row["seed_schedule_delta_per_case"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Feedback Operator Selection",
            "",
            "These rows expose whether feedback mutation actually selected operators aligned with semantic-family or semantic-signal targets, rather than only configuring those hooks.",
            "",
            "| target suite | variant | preset | feedback target keys | family targets | signal targets | affinity-hit cases | top selected operators | top semantic target keys |",
            "|---|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {feedback_target_key_count} | "
            "{feedback_semantic_family_target_count} | {feedback_semantic_signal_target_count} | "
            "{feedback_operator_affinity_hit_cases} ({feedback_operator_affinity_hit_rate}) | "
            "{top_feedback_selected_operators} | {top_feedback_semantic_target_keys} |".format(
                **{
                    **row,
                    "feedback_operator_affinity_hit_rate": _fmt_percent(row["feedback_operator_affinity_hit_rate"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Storage Efficiency",
            "",
            "Storage is measured from the compressed/uncompressed run log plus unique bug artifact directories referenced by the run rows. Use `evidence bytes/case` to compare logging and artifact overhead across variants.",
            "",
            "| target suite | variant | preset | runs | cases | run log bytes | artifact bytes | evidence bytes | evidence bytes/case |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {runs} | {cases} | {run_log_bytes} | {artifact_bytes} | {evidence_bytes} | {evidence_bytes_per_case} |".format(
                **{
                    **row,
                    "evidence_bytes_per_case": _fmt_float(row["evidence_bytes_per_case"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Stage Profiling",
            "",
            "Per-run stage timings come from the runner and are aggregated into run metadata for long-run health analysis and freeze-safe methodology reporting.",
            "",
            "| target suite | variant | preset | runs | cases | gen/mutate ms | backend ms | normalize ms | oracle ms | scheduler ms | logging ms | total ms/case | backend share | oracle share | scheduler share |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {runs} | {cases} | {stage_generate_mutate_avg_ms} | "
            "{stage_backend_execution_avg_ms} | {stage_normalize_avg_ms} | {stage_oracle_classification_avg_ms} | "
            "{stage_scheduler_feedback_avg_ms} | {stage_logging_artifact_avg_ms} | {stage_total_case_wall_avg_ms} | "
            "{stage_backend_execution_share} | {stage_oracle_classification_share} | {stage_scheduler_feedback_share} |".format(
                **{
                    **row,
                    "stage_generate_mutate_avg_ms": _fmt_float(row["stage_generate_mutate_avg_ms"]),
                    "stage_backend_execution_avg_ms": _fmt_float(row["stage_backend_execution_avg_ms"]),
                    "stage_normalize_avg_ms": _fmt_float(row["stage_normalize_avg_ms"]),
                    "stage_oracle_classification_avg_ms": _fmt_float(row["stage_oracle_classification_avg_ms"]),
                    "stage_scheduler_feedback_avg_ms": _fmt_float(row["stage_scheduler_feedback_avg_ms"]),
                    "stage_logging_artifact_avg_ms": _fmt_float(row["stage_logging_artifact_avg_ms"]),
                    "stage_total_case_wall_avg_ms": _fmt_float(row["stage_total_case_wall_avg_ms"]),
                    "stage_backend_execution_share": _fmt_percent(row["stage_backend_execution_share"]),
                    "stage_oracle_classification_share": _fmt_percent(row["stage_oracle_classification_share"]),
                    "stage_scheduler_feedback_share": _fmt_percent(row["stage_scheduler_feedback_share"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Semantic Scheduling",
            "",
            "These rows expose the configured semantic targets and discovery-bias hooks, plus the observed target/bias/family/signal hits recorded by the guidance layer during execution.",
            "",
            "| target suite | variant | preset | configured targets | configured discovery biases | matched-target cases | bias-hit cases | avg matched targets | avg bias hits | top matched targets | top bias hits | top semantic families | top semantic signals |",
            "|---|---|---|---|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {configured_guidance_targets} | {configured_discovery_biases} | "
            "{matched_semantic_target_cases} ({matched_semantic_target_case_rate}) | "
            "{discovery_bias_hit_cases} ({discovery_bias_hit_case_rate}) | "
            "{avg_matched_semantic_target_count} | {avg_discovery_bias_hit_count} | "
            "{top_matched_semantic_targets} | {top_discovery_bias_hits} | "
            "{top_observed_semantic_families} | {top_observed_semantic_signals} |".format(
                **{
                    **row,
                    "matched_semantic_target_case_rate": _fmt_percent(row["matched_semantic_target_case_rate"]),
                    "discovery_bias_hit_case_rate": _fmt_percent(row["discovery_bias_hit_case_rate"]),
                    "avg_matched_semantic_target_count": _fmt_float(row["avg_matched_semantic_target_count"]),
                    "avg_discovery_bias_hit_count": _fmt_float(row["avg_discovery_bias_hit_count"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Family Saturation Filter",
            "",
            "This pre-execution gate skips candidates predicted to reproduce dynamically saturated bug-family roots before spending backend execution time. Static known bug families are only downweighted after execution, not hard-filtered here.",
            "",
            "| target suite | variant | preset | runs | cases | filtered candidates | fallback candidates | filtered/case | fallback/case |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {runs} | {cases} | {family_saturation_filter_filtered_candidates} | "
            "{family_saturation_filter_fallback_candidates} | {family_saturation_filter_filtered_per_case} | "
            "{family_saturation_filter_fallback_per_case} |".format(
                **{
                    **row,
                    "family_saturation_filter_filtered_per_case": _fmt_float(
                        row["family_saturation_filter_filtered_per_case"]
                    ),
                    "family_saturation_filter_fallback_per_case": _fmt_float(
                        row["family_saturation_filter_fallback_per_case"]
                    ),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Preflight Integrity",
            "",
            "| target suite | variant | preset | runs | cases | invalid % | repaired % | fallback % |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {runs} | {cases} | {preflight_invalid_rate} | {preflight_repaired_rate} | {preflight_fallback_rate} |".format(
                **{
                    **row,
                    "preflight_invalid_rate": _fmt_percent(row["preflight_invalid_rate"]),
                    "preflight_repaired_rate": _fmt_percent(row["preflight_repaired_rate"]),
                    "preflight_fallback_rate": _fmt_percent(row["preflight_fallback_rate"]),
                }
            )
        )
    lines.extend(
        [
            "",
            "## Candidate Bug-Family Deduplication",
            "",
            "Families are conservatively grouped by `root_cause + suspicious_backends`; metamorphic relation failures are mapped to a co-occurring differential root for the same suspicious backend when one exists. Use this table for paper-level bug-family counts; use finding and case counts above for sensitivity and time-to-first measurements.",
            "",
            "| target suite | variant | preset | candidate bug families | top families |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| {target_suite} | {variant_label} | {preset} | {candidate_bug_families} | {top_candidate_bug_families} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Compare the reference variant `baseline` with `no_type_aware` to measure generator validity and useful behavior discovery.",
            "- Compare the reference variant `baseline` with `no_normalizer` to estimate false-positive pressure from representation differences.",
            "- Compare the reference variant `baseline` with `no_feedback` to measure whether corpus feedback improves new behavior discovery.",
            "- Compare the reference variant `baseline` with `metamorphic` to separate cross-engine differential bugs from single-engine relation violations.",
            "- Use `edge_float` separately from `common`; it studies boundary semantics rather than the default common subset.",
        ]
    )
    if manifest.get("schedule") == "adaptive":
        adaptive_summary = _adaptive_schedule_summary(rows)
        adaptive_final_arms = _adaptive_final_arm_rows(manifest.get("adaptive_state", []))
        lines.extend(
            [
                "",
                "## Adaptive Schedule",
                "",
                f"- Config: `{manifest.get('adaptive_config', {})}`",
                f"- Final arm state entries: {len(manifest.get('adaptive_state', []))}",
                f"- Global first candidate case: {_fmt_optional_int(adaptive_summary['global_first_candidate_case'])}",
                f"- Batches by suite: {adaptive_summary['batches_by_suite']}",
                f"- Cases by suite: {adaptive_summary['cases_by_suite']}",
                f"- Candidate cases by suite: {adaptive_summary['candidate_cases_by_suite']}",
            ]
        )
        if adaptive_final_arms:
            lines.extend(
                [
                    "",
                    "| arm | target suite | pulls | mean reward | reward signal | last reward | stale batches |",
                    "|---|---|---:|---:|---:|---:|---:|",
                ]
            )
            for arm in adaptive_final_arms:
                lines.append(
                    "| {arm_id} | {target_suite} | {pulls} | {mean_reward} | {reward_signal} | {last_reward} | {stale_batches} |".format(
                        arm_id=arm["arm_id"],
                        target_suite=arm["target_suite"],
                        pulls=_fmt_optional_int(arm["pulls"]),
                        mean_reward=_fmt_optional_float(arm["mean_reward"]),
                        reward_signal=_fmt_optional_float(arm["reward_signal"]),
                        last_reward=_fmt_optional_float(arm["last_reward"]),
                        stale_batches=_fmt_optional_int(arm["stale_batches"]),
                    )
                )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_suite",
                "preset",
                "matrix_id",
                "matrix_title",
                "comparison_group",
                "variant_id",
                "variant_title",
                "variant_label",
                "variant_group_id",
                "base_preset",
                "comparison_role",
                "canonical_comparison_role",
                "component_focus",
                "overlays",
                "semantic_focus_families",
                "semantic_focus_signals",
                "factors",
                "scope_kind",
                "oracle_profile",
                "rq_tags",
                "analysis_tags",
                "counts_as_real_bugs",
                "seed",
                "evidence_mode",
                "known_bug_id",
                "target_version",
                "batch_index",
                "schedule_arm_id",
                "scheduler_reward",
                "scheduler_reward_signal",
                "scheduler_mean_reward",
                "scheduler_last_reward",
                "scheduler_pulls",
                "scheduler_stale_batches",
                "enable_replay_bug",
                "replay_filter_enabled",
                "replay_filter_filtered_candidates",
                "replay_filter_fallback_candidates",
                "family_saturation_filter_enabled",
                "family_saturation_filter_filtered_candidates",
                "family_saturation_filter_fallback_candidates",
                "configured_guidance_targets",
                "configured_effective_guidance_targets",
                "configured_semantic_focus_families",
                "configured_semantic_focus_signals",
                "configured_discovery_biases",
                "cases",
                "run_log_bytes",
                "run_log_bytes_per_case",
                "artifact_bytes",
                "artifact_bytes_per_case",
                "evidence_bytes",
                "evidence_bytes_per_case",
                "bug_cases",
                "bug_rate",
                "candidate_bug_cases",
                "candidate_bug_case_rate",
                "first_finding_case_index",
                "first_candidate_bug_case_index",
                "first_candidate_bug_elapsed_s",
                "candidate_bug_discovery_auc",
                "findings",
                "unique_findings",
                "new_behavior_cases",
                "new_behavior_rate",
                "signal_new_behavior_cases",
                "signal_new_behavior_rate",
                "feedback_case_count",
                "feedback_mutation_cases",
                "feedback_mutation_case_rate",
                "feedback_target_key_count",
                "feedback_semantic_family_target_count",
                "feedback_semantic_signal_target_count",
                "feedback_operator_affinity_hit_cases",
                "feedback_operator_affinity_hit_rate",
                "feedback_selected_operator_count",
                "feedback_selected_operator_score_total",
                "feedback_selected_operator_score_avg",
                "top_feedback_selected_operators",
                "top_feedback_semantic_target_keys",
                "stored_in_feedback_corpus_cases",
                "feedback_corpus_store_rate",
                "quality_oracle_count",
                "quality_pass_count",
                "quality_fail_count",
                "quality_pass_rate",
                "quality_score_total",
                "quality_score_per_case",
                "source_reward_adjustment_total",
                "source_reward_adjustment_per_case",
                "guidance_reward_adjustment_total",
                "guidance_reward_adjustment_per_case",
                "seed_schedule_delta_total",
                "seed_schedule_delta_per_case",
                "productive_mutation_cases",
                "invalid_mutation_cases",
                "redundant_mutation_cases",
                "productive_mutation_rate",
                "invalid_mutation_rate",
                "feedback_finding_yield_cases",
                "feedback_new_behavior_yield_cases",
                "feedback_redundant_behavior_cases",
                "guided_productive_cases",
                "guided_target_miss_cases",
                "guided_redundant_cases",
                "guided_productive_rate",
                "guided_target_miss_rate",
                "preflight_repaired_cases",
                "preflight_fallback_cases",
                "preflight_invalid_cases",
                "preflight_repaired_rate",
                "preflight_fallback_rate",
                "preflight_invalid_rate",
                "elapsed_s",
                "throughput_cases_s",
                "stage_generate_mutate_total_ms",
                "stage_backend_execution_total_ms",
                "stage_normalize_total_ms",
                "stage_oracle_classification_total_ms",
                "stage_scheduler_feedback_total_ms",
                "stage_logging_artifact_total_ms",
                "stage_total_case_wall_ms",
                "stage_generate_mutate_avg_ms",
                "stage_backend_execution_avg_ms",
                "stage_normalize_avg_ms",
                "stage_oracle_classification_avg_ms",
                "stage_scheduler_feedback_avg_ms",
                "stage_logging_artifact_avg_ms",
                "stage_generate_mutate_share",
                "stage_backend_execution_share",
                "stage_normalize_share",
                "stage_oracle_classification_share",
                "stage_scheduler_feedback_share",
                "stage_logging_artifact_share",
                "avg_guidance_score",
                "avg_data_sensitivity",
                "avg_path_coverage_proxy",
                "avg_frontier_conformance",
                "avg_discovery_diversity_bonus",
                "avg_candidate_pool_diversity_bonus",
                "avg_discovery_stale_penalty",
                "avg_contribution_potential",
                "avg_candidate_count",
                "avg_contributing_candidate_count",
                "avg_pruned_candidate_count",
                "pruned_candidate_rate",
                "avg_feature_count",
                "avg_frontier_bucket_count",
                "avg_discovery_bucket_count",
                "matched_semantic_target_cases",
                "matched_semantic_target_case_rate",
                "discovery_bias_hit_cases",
                "discovery_bias_hit_case_rate",
                "avg_matched_semantic_target_count",
                "avg_discovery_bias_hit_count",
                "top_matched_semantic_targets",
                "top_discovery_bias_hits",
                "top_observed_semantic_families",
                "top_observed_semantic_signals",
                "top_root_causes",
                "top_finding_kinds",
                "top_triage_verdicts",
                "top_discovery_origins",
                "candidate_bug_families",
                "top_candidate_bug_families",
                "candidate_implementation_bug_count",
                "rewardable_candidate_implementation_bug_count",
                "issue_replay_candidate_bug_count",
                "known_saturated_candidate_bug_count",
                "documented_semantic_divergence_count",
                "expected_semantic_divergence_count",
                "semantic_divergence_needs_confirmation_count",
                "generator_false_positive_count",
                "normalizer_false_positive_count",
                "needs_manual_confirmation_count",
                "target_families",
                "backends",
                "run_file",
                "report",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with aggregate_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_suite",
                "preset",
                "matrix_id",
                "matrix_title",
                "comparison_group",
                "variant_id",
                "variant_title",
                "variant_label",
                "variant_group_id",
                "base_preset",
                "comparison_role",
                "canonical_comparison_role",
                "component_focus",
                "overlays",
                "semantic_focus_families",
                "semantic_focus_signals",
                "factors",
                "scope_kind",
                "oracle_profile",
                "rq_tags",
                "analysis_tags",
                "runs",
                "cases",
                "run_log_bytes",
                "run_log_bytes_per_case",
                "artifact_bytes",
                "artifact_bytes_per_case",
                "evidence_bytes",
                "evidence_bytes_per_case",
                "stage_generate_mutate_total_ms",
                "stage_backend_execution_total_ms",
                "stage_normalize_total_ms",
                "stage_oracle_classification_total_ms",
                "stage_scheduler_feedback_total_ms",
                "stage_logging_artifact_total_ms",
                "stage_total_case_wall_ms",
                "stage_generate_mutate_avg_ms",
                "stage_backend_execution_avg_ms",
                "stage_normalize_avg_ms",
                "stage_oracle_classification_avg_ms",
                "stage_scheduler_feedback_avg_ms",
                "stage_logging_artifact_avg_ms",
                "stage_total_case_wall_avg_ms",
                "stage_generate_mutate_share",
                "stage_backend_execution_share",
                "stage_normalize_share",
                "stage_oracle_classification_share",
                "stage_scheduler_feedback_share",
                "stage_logging_artifact_share",
                "findings",
                "candidate_implementation_bug_count",
                "rewardable_candidate_implementation_bug_count",
                "issue_replay_candidate_bug_count",
                "known_saturated_candidate_bug_count",
                "candidate_bug_cases",
                "candidate_bug_case_rate",
                "candidate_bug_cases_per_s",
                "median_first_candidate_bug_case_index",
                "median_first_candidate_bug_elapsed_s",
                "avg_candidate_bug_discovery_auc",
                "avg_scheduler_reward",
                "avg_scheduler_reward_signal",
                "avg_scheduler_mean_reward",
                "new_behavior_cases",
                "feedback_case_count",
                "feedback_mutation_cases",
                "feedback_mutation_case_rate",
                "feedback_target_key_count",
                "feedback_semantic_family_target_count",
                "feedback_semantic_signal_target_count",
                "feedback_operator_affinity_hit_cases",
                "feedback_operator_affinity_hit_rate",
                "feedback_selected_operator_count",
                "feedback_selected_operator_score_total",
                "feedback_selected_operator_score_avg",
                "top_feedback_selected_operators",
                "top_feedback_semantic_target_keys",
                "stored_in_feedback_corpus_cases",
                "feedback_corpus_store_rate",
                "quality_oracle_count",
                "quality_pass_count",
                "quality_fail_count",
                "quality_pass_rate",
                "quality_score_total",
                "quality_score_per_case",
                "source_reward_adjustment_total",
                "source_reward_adjustment_per_case",
                "guidance_reward_adjustment_total",
                "guidance_reward_adjustment_per_case",
                "seed_schedule_delta_total",
                "seed_schedule_delta_per_case",
                "productive_mutation_cases",
                "invalid_mutation_cases",
                "redundant_mutation_cases",
                "productive_mutation_rate",
                "invalid_mutation_rate",
                "feedback_finding_yield_cases",
                "feedback_new_behavior_yield_cases",
                "feedback_redundant_behavior_cases",
                "guided_productive_cases",
                "guided_target_miss_cases",
                "guided_redundant_cases",
                "guided_productive_rate",
                "guided_target_miss_rate",
                "preflight_repaired_cases",
                "preflight_fallback_cases",
                "preflight_invalid_cases",
                "preflight_repaired_rate",
                "preflight_fallback_rate",
                "preflight_invalid_rate",
                "family_saturation_filter_filtered_candidates",
                "family_saturation_filter_fallback_candidates",
                "family_saturation_filter_filtered_per_case",
                "family_saturation_filter_fallback_per_case",
                "configured_guidance_targets",
                "configured_effective_guidance_targets",
                "configured_semantic_focus_families",
                "configured_semantic_focus_signals",
                "configured_discovery_biases",
                "semantic_divergence_count",
                "false_positive_count",
                "new_behavior_cases",
                "avg_new_behavior_rate",
                "signal_new_behavior_cases",
                "avg_signal_new_behavior_rate",
                "avg_throughput_cases_s",
                "avg_data_sensitivity",
                "avg_path_coverage_proxy",
                "avg_discovery_diversity_bonus",
                "avg_candidate_pool_diversity_bonus",
                "avg_discovery_stale_penalty",
                "avg_discovery_bucket_count",
                "matched_semantic_target_cases",
                "matched_semantic_target_case_rate",
                "discovery_bias_hit_cases",
                "discovery_bias_hit_case_rate",
                "avg_matched_semantic_target_count",
                "avg_discovery_bias_hit_count",
                "top_matched_semantic_targets",
                "top_discovery_bias_hits",
                "top_observed_semantic_families",
                "top_observed_semantic_signals",
                "candidate_bug_families",
                "top_candidate_bug_families",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate_rows)

    aggregate_payload = _experiment_summary_aggregate_payload(
        manifest_file=manifest_file,
        manifest=manifest,
        experiment_meta=experiment_meta,
        run_rows=rows,
        variant_rows=aggregate_rows,
        aggregate_csv_path=aggregate_csv_path,
        summary_csv_path=csv_path,
        summary_markdown_path=md_path,
    )
    aggregate_json_path.write_text(
        json.dumps(aggregate_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return md_path, csv_path


def latest_run_file() -> Path:
    return latest_run_log_path()


def write_report(run_file: Path | None = None, csv_limit: int | None = None) -> tuple[Path, Path]:
    return write_run_report(run_file, csv_limit=csv_limit)


def latest_experiment_manifest() -> Path:
    return latest_experiment_manifest_path()


def write_experiment_summary(manifest_file: Path | None = None, *, refresh: bool = False) -> tuple[Path, Path]:
    return write_experiment_summary_report(manifest_file, refresh=refresh)


def _counter_summary(counter: Counter) -> str:
    return "; ".join(f"{key}:{count}" for key, count in counter.most_common(5)) or "none"


def _file_size_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _artifact_bytes_from_rows(rows: list[dict]) -> int:
    total = 0
    seen: set[Path] = set()
    for row in rows:
        bug_dir_text = str(row.get("bug_dir", "") or "").strip()
        if not bug_dir_text:
            continue
        path = Path(bug_dir_text)
        if not path.is_absolute() and not path.exists():
            path = Path.cwd() / path
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen:
            continue
        seen.add(key)
        total += _tree_size_bytes(path)
    return total


def _tree_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return _file_size_bytes(path)
    total = 0
    try:
        children = list(path.rglob("*"))
    except OSError:
        return 0
    for child in children:
        if child.is_file():
            total += _file_size_bytes(child)
    return total


def _parse_counter_summary(value: str) -> Counter:
    out: Counter = Counter()
    if not value or value == "none":
        return out
    for item in value.split("; "):
        if ":" not in item:
            continue
        key, count = item.rsplit(":", 1)
        try:
            out[key] += int(count)
        except ValueError:
            continue
    return out


def _refresh_summary_row_findings(row: dict) -> dict:
    refreshed = _recompute_differential_findings(row)
    if refreshed is None:
        return row
    case, normalized, findings, config, backends = refreshed
    annotate_findings(
        case,
        findings,
        normalized,
        row.get("raw_results", {}),
        config,
        backends,
    )
    out = dict(row)
    out["findings"] = [finding.to_dict() for finding in findings]
    return out


def _recompute_differential_findings(
    row: dict,
) -> tuple[Case, dict[str, NormalizedResult], list, dict, list[str]] | None:
    if row.get("normalized"):
        try:
            case = Case.from_dict(row["case"])
        except Exception:
            return None
        normalized = _normalized_from_mapping(row.get("normalized", {}))
        config = row.get("config", {})
        return case, normalized, evaluate_case(case, normalized), config, list(normalized)

    bug_dir_text = row.get("bug_dir", "")
    if not bug_dir_text:
        return None
    bug_dir = Path(bug_dir_text)
    if not bug_dir.exists():
        bug_dir = Path.cwd() / bug_dir_text
    case_path = bug_dir / "case.json"
    normalized_path = bug_dir / "normalized.json"
    if not case_path.exists() or not normalized_path.exists():
        return None
    case = Case.from_dict(load_json(case_path))
    normalized = _normalized_from_mapping(load_json(normalized_path))
    config_path = bug_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else ExperimentConfig().to_dict()
    return case, normalized, evaluate_case(case, normalized), config, list(normalized)


def _normalized_from_mapping(mapping: dict) -> dict[str, NormalizedResult]:
    return normalized_results_from_mapping(mapping)


def _candidate_issue_family_keys(
    findings: list[dict],
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> Counter:
    return reward_candidate_bug_family_keys(
        findings,
        known_saturated_bug_families=known_saturated_bug_families,
    )


def _is_candidate_issue_finding(finding: dict) -> bool:
    return reward_is_candidate_bug_finding(finding)


def _is_issue_replay_finding(finding: dict) -> bool:
    return reward_is_issue_replay_finding(finding)


def _is_known_saturated_candidate_issue_finding(
    finding: dict,
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return reward_is_known_saturated_candidate_bug_finding(finding, known_saturated_bug_families)


def _is_rewardable_candidate_issue_finding(
    finding: dict,
    known_saturated_bug_families: list[str] | tuple[str, ...] | None = None,
) -> bool:
    return reward_is_rewardable_candidate_bug_finding(finding, known_saturated_bug_families)


def _fmt_float(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def _fmt_percent(value) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return str(value)


def _fmt_optional_int(value) -> str:
    return "" if value is None else str(value)


def _fmt_optional_float(value) -> str:
    if value is None or value == "":
        return ""
    return _fmt_float(value)


def _fmt_optional_number(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _first_case_index(rows: list[dict], predicate) -> int | None:
    for idx, row in enumerate(rows):
        for finding in row.get("findings", []):
            if predicate(finding):
                return int(row.get("case_index", idx))
    return None


def _first_case_elapsed_s(rows: list[dict], predicate) -> float | None:
    for row in rows:
        for finding in row.get("findings", []):
            if predicate(finding):
                value = row.get("elapsed_s")
                return float(value) if value not in (None, "") else None
    return None


def _candidate_issue_discovery_auc(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    hits = [
        int(any(_is_candidate_issue_finding(finding) for finding in row.get("findings", [])))
        for row in rows
    ]
    total = sum(hits)
    if total == 0:
        return 0.0
    cumulative = 0
    area = 0
    for hit in hits:
        cumulative += hit
        area += cumulative
    return area / (len(rows) * total)


_candidate_bug_family_keys = _candidate_issue_family_keys
_is_candidate_bug_finding = _is_candidate_issue_finding
_is_known_saturated_candidate_bug_finding = _is_known_saturated_candidate_issue_finding
_is_rewardable_candidate_bug_finding = _is_rewardable_candidate_issue_finding
_candidate_bug_discovery_auc = _candidate_issue_discovery_auc


def _aggregate_experiment_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[experiment_row_variant_key(row)].append(row)

    out = []
    for (target_suite, comparison_group, matrix_id, variant_id, preset), items in sorted(grouped.items()):
        cases = sum(int(row["cases"]) for row in items)
        findings = sum(int(row["findings"]) for row in items)
        candidate_count = sum(int(row["candidate_implementation_bug_count"]) for row in items)
        rewardable_candidate_count = sum(
            int(row.get("rewardable_candidate_implementation_bug_count", 0) or 0) for row in items
        )
        issue_replay_candidate_count = sum(
            int(row.get("issue_replay_candidate_bug_count", 0) or 0) for row in items
        )
        known_saturated_candidate_count = sum(
            int(row.get("known_saturated_candidate_bug_count", 0) or 0) for row in items
        )
        family_counter = Counter()
        for row in items:
            family_counter.update(_parse_counter_summary(str(row.get("top_candidate_bug_families", ""))))
        candidate_bug_cases = sum(int(row["candidate_bug_cases"]) for row in items)
        semantic_divergence_count = sum(
            int(row["documented_semantic_divergence_count"])
            + int(row["expected_semantic_divergence_count"])
            + int(row["semantic_divergence_needs_confirmation_count"])
            for row in items
        )
        false_positive_count = sum(
            int(row["generator_false_positive_count"]) + int(row["normalizer_false_positive_count"])
            for row in items
        )
        run_log_bytes = sum(int(row.get("run_log_bytes", 0) or 0) for row in items)
        artifact_bytes = sum(int(row.get("artifact_bytes", 0) or 0) for row in items)
        evidence_bytes = run_log_bytes + artifact_bytes
        preflight_repaired_cases = sum(int(row.get("preflight_repaired_cases", 0) or 0) for row in items)
        preflight_fallback_cases = sum(int(row.get("preflight_fallback_cases", 0) or 0) for row in items)
        preflight_invalid_cases = sum(int(row.get("preflight_invalid_cases", 0) or 0) for row in items)
        family_saturation_filter_filtered_candidates = sum(
            int(row.get("family_saturation_filter_filtered_candidates", 0) or 0) for row in items
        )
        family_saturation_filter_fallback_candidates = sum(
            int(row.get("family_saturation_filter_fallback_candidates", 0) or 0) for row in items
        )
        feedback_case_count = sum(int(row.get("feedback_case_count", 0) or 0) for row in items)
        feedback_mutation_cases = sum(int(row.get("feedback_mutation_cases", 0) or 0) for row in items)
        stored_in_feedback_corpus_cases = sum(
            int(row.get("stored_in_feedback_corpus_cases", 0) or 0) for row in items
        )
        quality_oracle_count = sum(int(row.get("quality_oracle_count", 0) or 0) for row in items)
        quality_pass_count = sum(int(row.get("quality_pass_count", 0) or 0) for row in items)
        quality_fail_count = sum(int(row.get("quality_fail_count", 0) or 0) for row in items)
        quality_score_total = sum(float(row.get("quality_score_total", 0.0) or 0.0) for row in items)
        stage_generate_mutate_total_ms = sum(float(row.get("stage_generate_mutate_total_ms", 0.0) or 0.0) for row in items)
        stage_backend_execution_total_ms = sum(float(row.get("stage_backend_execution_total_ms", 0.0) or 0.0) for row in items)
        stage_normalize_total_ms = sum(float(row.get("stage_normalize_total_ms", 0.0) or 0.0) for row in items)
        stage_oracle_classification_total_ms = sum(
            float(row.get("stage_oracle_classification_total_ms", 0.0) or 0.0) for row in items
        )
        stage_scheduler_feedback_total_ms = sum(
            float(row.get("stage_scheduler_feedback_total_ms", 0.0) or 0.0) for row in items
        )
        stage_logging_artifact_total_ms = sum(
            float(row.get("stage_logging_artifact_total_ms", 0.0) or 0.0) for row in items
        )
        stage_total_case_wall_ms = sum(float(row.get("stage_total_case_wall_ms", 0.0) or 0.0) for row in items)
        source_reward_adjustment_total = sum(
            float(row.get("source_reward_adjustment_total", 0.0) or 0.0) for row in items
        )
        guidance_reward_adjustment_total = sum(
            float(row.get("guidance_reward_adjustment_total", 0.0) or 0.0) for row in items
        )
        seed_schedule_delta_total = sum(float(row.get("seed_schedule_delta_total", 0.0) or 0.0) for row in items)
        productive_mutation_cases = sum(int(row.get("productive_mutation_cases", 0) or 0) for row in items)
        invalid_mutation_cases = sum(int(row.get("invalid_mutation_cases", 0) or 0) for row in items)
        redundant_mutation_cases = sum(int(row.get("redundant_mutation_cases", 0) or 0) for row in items)
        feedback_finding_yield_cases = sum(
            int(row.get("feedback_finding_yield_cases", 0) or 0) for row in items
        )
        feedback_new_behavior_yield_cases = sum(
            int(row.get("feedback_new_behavior_yield_cases", 0) or 0) for row in items
        )
        feedback_redundant_behavior_cases = sum(
            int(row.get("feedback_redundant_behavior_cases", 0) or 0) for row in items
        )
        guided_productive_cases = sum(int(row.get("guided_productive_cases", 0) or 0) for row in items)
        guided_target_miss_cases = sum(int(row.get("guided_target_miss_cases", 0) or 0) for row in items)
        guided_redundant_cases = sum(int(row.get("guided_redundant_cases", 0) or 0) for row in items)
        new_behavior_cases = sum(int(row.get("new_behavior_cases", 0) or 0) for row in items)
        signal_new_behavior_cases = sum(int(row.get("signal_new_behavior_cases", 0) or 0) for row in items)
        avg_throughput = _avg_value(items, "throughput_cases_s")
        candidate_bug_case_rate = candidate_bug_cases / cases if cases else 0.0
        matrix_title = _first_nonempty_row_string(items, "matrix_title")
        variant_title = _first_nonempty_row_string(items, "variant_title")
        base_preset = _first_nonempty_row_string(items, "base_preset")
        comparison_role = _first_nonempty_row_string(items, "comparison_role")
        canonical_role = _first_nonempty_row_string(items, "canonical_comparison_role")
        component_focus = _first_nonempty_row_string(items, "component_focus")
        scope_kind = _first_nonempty_row_string(items, "scope_kind")
        oracle_profile = _first_nonempty_row_string(items, "oracle_profile")
        overlays = _csv_set_union(items, "overlays")
        semantic_focus_families = _csv_set_union(items, "semantic_focus_families")
        semantic_focus_signals = _csv_set_union(items, "semantic_focus_signals")
        merged_factors = _merged_row_factor_map(items)
        rq_tags = _csv_set_union(items, "rq_tags")
        analysis_tags = _csv_set_union(items, "analysis_tags")
        configured_guidance_targets = _csv_set_union(items, "configured_guidance_targets")
        configured_effective_guidance_targets = _csv_set_union(items, "configured_effective_guidance_targets")
        configured_semantic_focus_families = _csv_set_union(items, "configured_semantic_focus_families")
        configured_semantic_focus_signals = _csv_set_union(items, "configured_semantic_focus_signals")
        configured_discovery_biases = [
            item for item in _csv_set_union(items, "configured_discovery_biases", delimiter=";") if item != "none"
        ]
        matched_target_counter = Counter()
        discovery_bias_counter = Counter()
        semantic_family_counter = Counter()
        semantic_signal_counter = Counter()
        feedback_selected_operator_counter = Counter()
        feedback_semantic_target_key_counter = Counter()
        for row in items:
            matched_target_counter.update(_parse_counter_summary(str(row.get("top_matched_semantic_targets", ""))))
            discovery_bias_counter.update(_parse_counter_summary(str(row.get("top_discovery_bias_hits", ""))))
            semantic_family_counter.update(_parse_counter_summary(str(row.get("top_observed_semantic_families", ""))))
            semantic_signal_counter.update(_parse_counter_summary(str(row.get("top_observed_semantic_signals", ""))))
            feedback_selected_operator_counter.update(
                _parse_counter_summary(str(row.get("top_feedback_selected_operators", "")))
            )
            feedback_semantic_target_key_counter.update(
                _parse_counter_summary(str(row.get("top_feedback_semantic_target_keys", "")))
            )
        out.append(
            {
                "target_suite": target_suite,
                "preset": preset,
                "matrix_id": matrix_id,
                "matrix_title": matrix_title,
                "comparison_group": comparison_group,
                "variant_id": variant_id or experiment_row_variant_id(items[0]),
                "variant_title": variant_title,
                "variant_label": experiment_row_variant_label(items[0]),
                "variant_group_id": "|".join(experiment_row_group_id(items[0])),
                "base_preset": base_preset,
                "comparison_role": comparison_role,
                "canonical_comparison_role": canonical_role,
                "component_focus": component_focus,
                "overlays": ",".join(overlays),
                "semantic_focus_families": ",".join(semantic_focus_families),
                "semantic_focus_signals": ",".join(semantic_focus_signals),
                "factors": json.dumps(merged_factors, ensure_ascii=False, sort_keys=True),
                "scope_kind": scope_kind,
                "oracle_profile": oracle_profile,
                "rq_tags": ",".join(rq_tags),
                "analysis_tags": ",".join(analysis_tags),
                "runs": len(items),
                "cases": cases,
                "run_log_bytes": run_log_bytes,
                "run_log_bytes_per_case": run_log_bytes / cases if cases else 0.0,
                "artifact_bytes": artifact_bytes,
                "artifact_bytes_per_case": artifact_bytes / cases if cases else 0.0,
                "evidence_bytes": evidence_bytes,
                "evidence_bytes_per_case": evidence_bytes / cases if cases else 0.0,
                "stage_generate_mutate_total_ms": stage_generate_mutate_total_ms,
                "stage_backend_execution_total_ms": stage_backend_execution_total_ms,
                "stage_normalize_total_ms": stage_normalize_total_ms,
                "stage_oracle_classification_total_ms": stage_oracle_classification_total_ms,
                "stage_scheduler_feedback_total_ms": stage_scheduler_feedback_total_ms,
                "stage_logging_artifact_total_ms": stage_logging_artifact_total_ms,
                "stage_total_case_wall_ms": stage_total_case_wall_ms,
                "stage_generate_mutate_avg_ms": stage_generate_mutate_total_ms / cases if cases else 0.0,
                "stage_backend_execution_avg_ms": stage_backend_execution_total_ms / cases if cases else 0.0,
                "stage_normalize_avg_ms": stage_normalize_total_ms / cases if cases else 0.0,
                "stage_oracle_classification_avg_ms": (
                    stage_oracle_classification_total_ms / cases if cases else 0.0
                ),
                "stage_scheduler_feedback_avg_ms": stage_scheduler_feedback_total_ms / cases if cases else 0.0,
                "stage_logging_artifact_avg_ms": stage_logging_artifact_total_ms / cases if cases else 0.0,
                "stage_total_case_wall_avg_ms": stage_total_case_wall_ms / cases if cases else 0.0,
                "stage_generate_mutate_share": (
                    stage_generate_mutate_total_ms / stage_total_case_wall_ms if stage_total_case_wall_ms else 0.0
                ),
                "stage_backend_execution_share": (
                    stage_backend_execution_total_ms / stage_total_case_wall_ms if stage_total_case_wall_ms else 0.0
                ),
                "stage_normalize_share": (
                    stage_normalize_total_ms / stage_total_case_wall_ms if stage_total_case_wall_ms else 0.0
                ),
                "stage_oracle_classification_share": (
                    stage_oracle_classification_total_ms / stage_total_case_wall_ms if stage_total_case_wall_ms else 0.0
                ),
                "stage_scheduler_feedback_share": (
                    stage_scheduler_feedback_total_ms / stage_total_case_wall_ms if stage_total_case_wall_ms else 0.0
                ),
                "stage_logging_artifact_share": (
                    stage_logging_artifact_total_ms / stage_total_case_wall_ms if stage_total_case_wall_ms else 0.0
                ),
                "findings": findings,
                "candidate_implementation_bug_count": candidate_count,
                "rewardable_candidate_implementation_bug_count": rewardable_candidate_count,
                "issue_replay_candidate_bug_count": issue_replay_candidate_count,
                "known_saturated_candidate_bug_count": known_saturated_candidate_count,
                "candidate_bug_families": len(family_counter),
                "top_candidate_bug_families": _counter_summary(family_counter),
                "candidate_bug_cases": candidate_bug_cases,
                "candidate_bug_case_rate": candidate_bug_case_rate,
                "candidate_bug_cases_per_s": candidate_bug_case_rate * avg_throughput,
                "median_first_candidate_bug_case_index": _median_optional_int(
                    row["first_candidate_bug_case_index"] for row in items
                ),
                "median_first_candidate_bug_elapsed_s": _median_optional_float(
                    row.get("first_candidate_bug_elapsed_s") for row in items
                ),
                "avg_candidate_bug_discovery_auc": _avg_value(items, "candidate_bug_discovery_auc"),
                "avg_scheduler_reward": _avg_value(items, "scheduler_reward"),
                "avg_scheduler_reward_signal": _avg_value(items, "scheduler_reward_signal"),
                "avg_scheduler_mean_reward": _avg_value(items, "scheduler_mean_reward"),
                "feedback_case_count": feedback_case_count,
                "feedback_mutation_cases": feedback_mutation_cases,
                "feedback_mutation_case_rate": feedback_mutation_cases / cases if cases else 0.0,
                "feedback_target_key_count": sum(int(row.get("feedback_target_key_count", 0) or 0) for row in items),
                "feedback_semantic_family_target_count": sum(
                    int(row.get("feedback_semantic_family_target_count", 0) or 0) for row in items
                ),
                "feedback_semantic_signal_target_count": sum(
                    int(row.get("feedback_semantic_signal_target_count", 0) or 0) for row in items
                ),
                "feedback_operator_affinity_hit_cases": sum(
                    int(row.get("feedback_operator_affinity_hit_cases", 0) or 0) for row in items
                ),
                "feedback_operator_affinity_hit_rate": (
                    sum(int(row.get("feedback_operator_affinity_hit_cases", 0) or 0) for row in items) / cases
                    if cases
                    else 0.0
                ),
                "feedback_selected_operator_count": sum(
                    int(row.get("feedback_selected_operator_count", 0) or 0) for row in items
                ),
                "feedback_selected_operator_score_total": sum(
                    float(row.get("feedback_selected_operator_score_total", 0.0) or 0.0) for row in items
                ),
                "feedback_selected_operator_score_avg": (
                    sum(float(row.get("feedback_selected_operator_score_total", 0.0) or 0.0) for row in items)
                    / sum(int(row.get("feedback_selected_operator_count", 0) or 0) for row in items)
                    if sum(int(row.get("feedback_selected_operator_count", 0) or 0) for row in items)
                    else 0.0
                ),
                "top_feedback_selected_operators": _counter_summary_limited(feedback_selected_operator_counter),
                "top_feedback_semantic_target_keys": _counter_summary_limited(feedback_semantic_target_key_counter),
                "stored_in_feedback_corpus_cases": stored_in_feedback_corpus_cases,
                "feedback_corpus_store_rate": stored_in_feedback_corpus_cases / cases if cases else 0.0,
                "quality_oracle_count": quality_oracle_count,
                "quality_pass_count": quality_pass_count,
                "quality_fail_count": quality_fail_count,
                "quality_pass_rate": quality_pass_count / quality_oracle_count if quality_oracle_count else 0.0,
                "quality_score_total": quality_score_total,
                "quality_score_per_case": quality_score_total / cases if cases else 0.0,
                "source_reward_adjustment_total": source_reward_adjustment_total,
                "source_reward_adjustment_per_case": source_reward_adjustment_total / cases if cases else 0.0,
                "guidance_reward_adjustment_total": guidance_reward_adjustment_total,
                "guidance_reward_adjustment_per_case": (
                    guidance_reward_adjustment_total / cases if cases else 0.0
                ),
                "seed_schedule_delta_total": seed_schedule_delta_total,
                "seed_schedule_delta_per_case": seed_schedule_delta_total / cases if cases else 0.0,
                "productive_mutation_cases": productive_mutation_cases,
                "invalid_mutation_cases": invalid_mutation_cases,
                "redundant_mutation_cases": redundant_mutation_cases,
                "productive_mutation_rate": (
                    productive_mutation_cases / feedback_mutation_cases if feedback_mutation_cases else 0.0
                ),
                "invalid_mutation_rate": (
                    invalid_mutation_cases / feedback_mutation_cases if feedback_mutation_cases else 0.0
                ),
                "feedback_finding_yield_cases": feedback_finding_yield_cases,
                "feedback_new_behavior_yield_cases": feedback_new_behavior_yield_cases,
                "feedback_redundant_behavior_cases": feedback_redundant_behavior_cases,
                "guided_productive_cases": guided_productive_cases,
                "guided_target_miss_cases": guided_target_miss_cases,
                "guided_redundant_cases": guided_redundant_cases,
                "guided_productive_rate": guided_productive_cases / cases if cases else 0.0,
                "guided_target_miss_rate": guided_target_miss_cases / cases if cases else 0.0,
                "new_behavior_cases": new_behavior_cases,
                "signal_new_behavior_cases": signal_new_behavior_cases,
                "preflight_repaired_cases": preflight_repaired_cases,
                "preflight_fallback_cases": preflight_fallback_cases,
                "preflight_invalid_cases": preflight_invalid_cases,
                "preflight_repaired_rate": preflight_repaired_cases / cases if cases else 0.0,
                "preflight_fallback_rate": preflight_fallback_cases / cases if cases else 0.0,
                "preflight_invalid_rate": preflight_invalid_cases / cases if cases else 0.0,
                "family_saturation_filter_filtered_candidates": family_saturation_filter_filtered_candidates,
                "family_saturation_filter_fallback_candidates": family_saturation_filter_fallback_candidates,
                "family_saturation_filter_filtered_per_case": (
                    family_saturation_filter_filtered_candidates / cases if cases else 0.0
                ),
                "family_saturation_filter_fallback_per_case": (
                    family_saturation_filter_fallback_candidates / cases if cases else 0.0
                ),
                "configured_guidance_targets": ",".join(configured_guidance_targets),
                "configured_effective_guidance_targets": ",".join(configured_effective_guidance_targets),
                "configured_semantic_focus_families": ",".join(configured_semantic_focus_families),
                "configured_semantic_focus_signals": ",".join(configured_semantic_focus_signals),
                "configured_discovery_biases": "; ".join(configured_discovery_biases) if configured_discovery_biases else "none",
                "semantic_divergence_count": semantic_divergence_count,
                "false_positive_count": false_positive_count,
                "avg_new_behavior_rate": _avg_value(items, "new_behavior_rate"),
                "avg_signal_new_behavior_rate": _avg_value(items, "signal_new_behavior_rate"),
                "avg_throughput_cases_s": avg_throughput,
                "avg_data_sensitivity": _avg_value(items, "avg_data_sensitivity"),
                "avg_path_coverage_proxy": _avg_value(items, "avg_path_coverage_proxy"),
                "avg_discovery_diversity_bonus": _avg_value(items, "avg_discovery_diversity_bonus"),
                "avg_candidate_pool_diversity_bonus": _avg_value(items, "avg_candidate_pool_diversity_bonus"),
                "avg_discovery_stale_penalty": _avg_value(items, "avg_discovery_stale_penalty"),
                "avg_discovery_bucket_count": _avg_value(items, "avg_discovery_bucket_count"),
                "matched_semantic_target_cases": sum(int(row.get("matched_semantic_target_cases", 0) or 0) for row in items),
                "matched_semantic_target_case_rate": (
                    sum(int(row.get("matched_semantic_target_cases", 0) or 0) for row in items) / cases if cases else 0.0
                ),
                "discovery_bias_hit_cases": sum(int(row.get("discovery_bias_hit_cases", 0) or 0) for row in items),
                "discovery_bias_hit_case_rate": (
                    sum(int(row.get("discovery_bias_hit_cases", 0) or 0) for row in items) / cases if cases else 0.0
                ),
                "avg_matched_semantic_target_count": _avg_value(items, "avg_matched_semantic_target_count"),
                "avg_discovery_bias_hit_count": _avg_value(items, "avg_discovery_bias_hit_count"),
                "top_matched_semantic_targets": _counter_summary_limited(matched_target_counter),
                "top_discovery_bias_hits": _counter_summary_limited(discovery_bias_counter),
                "top_observed_semantic_families": _counter_summary_limited(semantic_family_counter),
                "top_observed_semantic_signals": _counter_summary_limited(semantic_signal_counter),
            }
        )
    return out


def _experiment_summary_aggregate_payload(
    *,
    manifest_file: Path,
    manifest: dict[str, Any],
    experiment_meta: dict[str, Any],
    run_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    aggregate_csv_path: Path,
    summary_csv_path: Path,
    summary_markdown_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": "experiment-summary-aggregates-v1",
        "manifest_file": str(manifest_file),
        "summary_markdown": str(summary_markdown_path),
        "summary_csv": str(summary_csv_path),
        "aggregate_csv": str(aggregate_csv_path),
        "experiment_meta": experiment_meta,
        "run_count": len(run_rows),
        "variant_count": len(variant_rows),
        "run_rows": run_rows,
        "variant_rows": variant_rows,
        "by_target_suite": _group_run_rows(
            run_rows,
            group_type="target_suite",
            memberships=lambda row: [("target_suite", row_string_value(row, "target_suite"))],
        ),
        "by_matrix": _group_run_rows(
            run_rows,
            group_type="matrix",
            memberships=lambda row: [("matrix", row_string_value(row, "matrix_id"))],
        ),
        "by_scope_kind": _group_run_rows(
            run_rows,
            group_type="scope_kind",
            memberships=lambda row: [("scope_kind", row_string_value(row, "scope_kind"))],
        ),
        "by_oracle_profile": _group_run_rows(
            run_rows,
            group_type="oracle_profile",
            memberships=lambda row: [("oracle_profile", row_string_value(row, "oracle_profile"))],
        ),
        "by_rq": _group_run_rows(
            run_rows,
            group_type="rq_tag",
            memberships=lambda row: [("rq_tag", tag) for tag in row_string_list(row, "rq_tags")],
        ),
        "by_factor": _group_run_rows(
            run_rows,
            group_type="factor",
            memberships=_factor_group_memberships,
        ),
        "analysis_tags": sorted(
            {
                tag
                for row in run_rows
                for tag in row_string_list(row, "analysis_tags")
            }
        ),
        "evidence_mode": str(manifest.get("evidence_mode", "") or ""),
    }


def _group_run_rows(
    rows: list[dict[str, Any]],
    *,
    group_type: str,
    memberships,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for key, value in memberships(row):
            normalized_value = str(value or "").strip()
            if not normalized_value:
                continue
            grouped[(key, normalized_value)].append(row)
    summaries = [
        _group_run_summary(items, group_type=group_type, group_key=key, group_value=value)
        for (key, value), items in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    return summaries


def _factor_group_memberships(row: dict[str, Any]) -> list[tuple[str, str]]:
    memberships: list[tuple[str, str]] = []
    for factor_name, factor_value in sorted(row_factor_map(row).items()):
        memberships.append((factor_name, _factor_group_value(factor_value)))
    return memberships


def _factor_group_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _group_run_summary(
    rows: list[dict[str, Any]],
    *,
    group_type: str,
    group_key: str,
    group_value: str,
) -> dict[str, Any]:
    cases = sum(int(row.get("cases", 0) or 0) for row in rows)
    findings = sum(int(row.get("findings", 0) or 0) for row in rows)
    candidate_bug_cases = sum(int(row.get("candidate_bug_cases", 0) or 0) for row in rows)
    candidate_implementation_bug_count = sum(
        int(row.get("candidate_implementation_bug_count", 0) or 0) for row in rows
    )
    rewardable_candidate_count = sum(
        int(row.get("rewardable_candidate_implementation_bug_count", 0) or 0) for row in rows
    )
    issue_replay_candidate_count = sum(
        int(row.get("issue_replay_candidate_bug_count", 0) or 0) for row in rows
    )
    known_saturated_candidate_count = sum(
        int(row.get("known_saturated_candidate_bug_count", 0) or 0) for row in rows
    )
    semantic_divergence_count = sum(
        int(row.get("documented_semantic_divergence_count", 0) or 0)
        + int(row.get("expected_semantic_divergence_count", 0) or 0)
        + int(row.get("semantic_divergence_needs_confirmation_count", 0) or 0)
        for row in rows
    )
    false_positive_count = sum(
        int(row.get("generator_false_positive_count", 0) or 0)
        + int(row.get("normalizer_false_positive_count", 0) or 0)
        for row in rows
    )
    run_log_bytes = sum(int(row.get("run_log_bytes", 0) or 0) for row in rows)
    artifact_bytes = sum(int(row.get("artifact_bytes", 0) or 0) for row in rows)
    evidence_bytes = run_log_bytes + artifact_bytes
    elapsed_s = sum(float(row.get("elapsed_s", 0.0) or 0.0) for row in rows)
    stage_totals = {
        "generate_mutate_ms": sum(float(row.get("stage_generate_mutate_total_ms", 0.0) or 0.0) for row in rows),
        "backend_execution_ms": sum(float(row.get("stage_backend_execution_total_ms", 0.0) or 0.0) for row in rows),
        "normalize_ms": sum(float(row.get("stage_normalize_total_ms", 0.0) or 0.0) for row in rows),
        "oracle_classification_ms": sum(
            float(row.get("stage_oracle_classification_total_ms", 0.0) or 0.0) for row in rows
        ),
        "scheduler_feedback_ms": sum(
            float(row.get("stage_scheduler_feedback_total_ms", 0.0) or 0.0) for row in rows
        ),
        "logging_artifact_ms": sum(float(row.get("stage_logging_artifact_total_ms", 0.0) or 0.0) for row in rows),
        "total_case_wall_ms": sum(float(row.get("stage_total_case_wall_ms", 0.0) or 0.0) for row in rows),
    }
    total_stage_wall_ms = float(stage_totals["total_case_wall_ms"] or 0.0)
    stage_avg = {
        key: (value / cases if cases else 0.0)
        for key, value in stage_totals.items()
    }
    stage_share = {
        key: (value / total_stage_wall_ms if total_stage_wall_ms else 0.0)
        for key, value in stage_totals.items()
    }
    factor_values: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        for factor_name, factor_value in row_factor_map(row).items():
            if factor_value not in factor_values[factor_name]:
                factor_values[factor_name].append(factor_value)
    summary = {
        "group_type": group_type,
        "group_key": group_key,
        "group_value": group_value,
        "runs": len(rows),
        "cases": cases,
        "findings": findings,
        "candidate_implementation_bug_count": candidate_implementation_bug_count,
        "rewardable_candidate_implementation_bug_count": rewardable_candidate_count,
        "issue_replay_candidate_bug_count": issue_replay_candidate_count,
        "known_saturated_candidate_bug_count": known_saturated_candidate_count,
        "candidate_bug_cases": candidate_bug_cases,
        "candidate_bug_case_rate": candidate_bug_cases / cases if cases else 0.0,
        "candidate_bug_cases_per_s": candidate_bug_cases / elapsed_s if elapsed_s else 0.0,
        "avg_candidate_bug_discovery_auc": _avg_value(rows, "candidate_bug_discovery_auc"),
        "semantic_divergence_count": semantic_divergence_count,
        "false_positive_count": false_positive_count,
        "run_log_bytes": run_log_bytes,
        "artifact_bytes": artifact_bytes,
        "evidence_bytes": evidence_bytes,
        "evidence_bytes_per_case": evidence_bytes / cases if cases else 0.0,
        "avg_throughput_cases_s": _avg_value(rows, "throughput_cases_s"),
        "avg_new_behavior_rate": _avg_value(rows, "new_behavior_rate"),
        "avg_signal_new_behavior_rate": _avg_value(rows, "signal_new_behavior_rate"),
        "target_suites": _csv_set_union(rows, "target_suite"),
        "matrix_ids": _csv_set_union(rows, "matrix_id"),
        "comparison_groups": _csv_set_union(rows, "comparison_group"),
        "variant_ids": _csv_set_union(rows, "variant_id"),
        "presets": _csv_set_union(rows, "preset"),
        "scope_kinds": _csv_set_union(rows, "scope_kind"),
        "oracle_profiles": _csv_set_union(rows, "oracle_profile"),
        "rq_tags": _csv_set_union(rows, "rq_tags"),
        "analysis_tags": _csv_set_union(rows, "analysis_tags"),
        "semantic_focus_families": _csv_set_union(rows, "semantic_focus_families"),
        "semantic_focus_signals": _csv_set_union(rows, "semantic_focus_signals"),
        "configured_effective_guidance_targets": _csv_set_union(rows, "configured_effective_guidance_targets"),
        "configured_discovery_biases": _csv_set_union(rows, "configured_discovery_biases", delimiter=";"),
        "factor_values": {name: values for name, values in sorted(factor_values.items())},
        "stage_profile": {
            "totals_ms": stage_totals,
            "avg_ms_per_case": stage_avg,
            "share_of_total": stage_share,
        },
    }
    if group_type == "factor":
        summary["factor_name"] = group_key
        summary["factor_value"] = json.loads(group_value) if group_value.startswith(("{", "[", "\"")) else (
            True if group_value == "true" else False if group_value == "false" else group_value
        )
    return summary


def _adaptive_schedule_summary(rows: list[dict]) -> dict[str, object]:
    batches_by_suite = Counter()
    cases_by_suite = Counter()
    candidate_cases_by_suite = Counter()
    global_first_candidate_case: int | None = None
    offset = 0
    for row in rows:
        suite = str(row.get("target_suite", "unknown"))
        batches_by_suite[suite] += 1
        cases = int(row.get("cases", 0) or 0)
        cases_by_suite[suite] += cases
        candidate_cases = int(row.get("candidate_bug_cases", 0) or 0)
        candidate_cases_by_suite[suite] += candidate_cases
        first_candidate = row.get("first_candidate_bug_case_index")
        if global_first_candidate_case is None and first_candidate not in (None, ""):
            global_first_candidate_case = offset + int(first_candidate) + 1
        offset += cases
    return {
        "global_first_candidate_case": global_first_candidate_case,
        "batches_by_suite": _counter_summary(batches_by_suite),
        "cases_by_suite": _counter_summary(cases_by_suite),
        "candidate_cases_by_suite": _counter_summary(candidate_cases_by_suite),
    }


def _adaptive_state_by_arm(adaptive_state: list[dict]) -> dict[str, dict]:
    by_arm: dict[str, dict] = {}
    for item in adaptive_state:
        if not isinstance(item, dict):
            continue
        arm_id = str(item.get("arm_id", "") or "")
        if not arm_id:
            continue
        by_arm[arm_id] = item
    return by_arm


def _adaptive_final_arm_rows(adaptive_state: list[dict], *, limit: int = 8) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in adaptive_state:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "arm_id": str(item.get("arm_id", "") or ""),
                "target_suite": str(item.get("target_suite", "") or ""),
                "pulls": item.get("pulls", ""),
                "mean_reward": item.get("mean_reward", ""),
                "reward_signal": item.get("reward_signal", ""),
                "last_reward": item.get("last_reward", ""),
                "stale_batches": item.get("stale_batches", ""),
            }
        )
    rows = [row for row in rows if row["arm_id"]]
    rows.sort(
        key=lambda row: (
            -(_float_or_none(row.get("reward_signal")) or 0.0),
            -(_float_or_none(row.get("mean_reward")) or 0.0),
            -(_float_or_none(row.get("pulls")) or 0.0),
            str(row.get("arm_id", "")),
        )
    )
    return rows[:limit]


def _avg_value(rows: list[dict], key: str) -> float:
    values = [_float_or_none(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0.0


def _csv_set_union(rows: list[dict], key: str, *, delimiter: str = ",") -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for item in row_string_list(row, key, delimiter=delimiter):
            if not item or item in seen:
                continue
            seen.add(item)
            values.append(item)
    return values


def _first_nonempty_row_string(rows: list[dict], key: str) -> str:
    for row in rows:
        value = row_string_value(row, key)
        if value:
            return value
    return ""


def _merged_row_factor_map(rows: list[dict]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for row in rows:
        merged.update(row_factor_map(row))
    return merged


def _median_optional_int(values) -> float | None:
    ints = sorted(int(value) for value in values if value is not None and value != "")
    if not ints:
        return None
    middle = len(ints) // 2
    if len(ints) % 2 == 1:
        return float(ints[middle])
    return (ints[middle - 1] + ints[middle]) / 2.0


def _median_optional_float(values) -> float | None:
    floats = sorted(float(value) for value in values if value is not None and value != "")
    if not floats:
        return None
    middle = len(floats) // 2
    if len(floats) % 2 == 1:
        return floats[middle]
    return (floats[middle - 1] + floats[middle]) / 2.0


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _guidance_metrics(rows: list[dict]) -> dict[str, float]:
    guidance_rows = [row.get("guidance", {}) for row in rows if row.get("guidance")]
    matched_target_counter: Counter[str] = Counter()
    discovery_bias_counter: Counter[str] = Counter()
    semantic_family_counter: Counter[str] = Counter()
    semantic_signal_counter: Counter[str] = Counter()
    matched_semantic_target_cases = 0
    discovery_bias_hit_cases = 0
    for row in guidance_rows:
        matched_targets = _string_list(row.get("matched_semantic_targets", row.get("matched_targets", [])))
        bias_hits = _string_list(row.get("discovery_bias_hits", []))
        features = _string_list(row.get("features", []))
        if matched_targets:
            matched_semantic_target_cases += 1
            matched_target_counter.update(matched_targets)
        if bias_hits:
            discovery_bias_hit_cases += 1
            discovery_bias_counter.update(bias_hits)
        semantic_family_counter.update(
            feature.split(":", 1)[1]
            for feature in features
            if feature.startswith("semantic_family:")
        )
        semantic_signal_counter.update(
            feature.split(":", 1)[1]
            for feature in features
            if feature.startswith("semantic_signal:")
        )
    return {
        "avg_guidance_score": _avg_metric(guidance_rows, "score"),
        "avg_data_sensitivity": _avg_metric(guidance_rows, "data_sensitivity"),
        "avg_path_coverage_proxy": _avg_metric(guidance_rows, "path_coverage_proxy"),
        "avg_frontier_conformance": _avg_metric(guidance_rows, "frontier_conformance"),
        "avg_discovery_diversity_bonus": _avg_metric(guidance_rows, "discovery_diversity_bonus"),
        "avg_candidate_pool_diversity_bonus": _avg_metric(guidance_rows, "candidate_pool_diversity_bonus"),
        "avg_discovery_stale_penalty": _avg_metric(guidance_rows, "discovery_stale_penalty"),
        "avg_contribution_potential": _avg_metric(guidance_rows, "contribution_potential"),
        "avg_candidate_count": _avg_metric(guidance_rows, "candidate_count"),
        "avg_contributing_candidate_count": _avg_metric(guidance_rows, "contributing_candidate_count"),
        "avg_pruned_candidate_count": _avg_metric(guidance_rows, "pruned_candidate_count"),
        "pruned_candidate_rate": _candidate_rate(guidance_rows, "pruned_candidate_count"),
        "avg_feature_count": _avg_metric(guidance_rows, "feature_count"),
        "avg_frontier_bucket_count": _avg_metric(guidance_rows, "frontier_bucket_count"),
        "avg_discovery_bucket_count": _avg_metric(guidance_rows, "discovery_bucket_count"),
        "matched_semantic_target_cases": matched_semantic_target_cases,
        "matched_semantic_target_case_rate": matched_semantic_target_cases / len(rows) if rows else 0.0,
        "discovery_bias_hit_cases": discovery_bias_hit_cases,
        "discovery_bias_hit_case_rate": discovery_bias_hit_cases / len(rows) if rows else 0.0,
        "avg_matched_semantic_target_count": _avg_metric(guidance_rows, "matched_semantic_target_count"),
        "avg_discovery_bias_hit_count": _avg_metric(guidance_rows, "discovery_bias_hit_count"),
        "top_matched_semantic_targets": _counter_summary_limited(matched_target_counter),
        "top_discovery_bias_hits": _counter_summary_limited(discovery_bias_counter),
        "top_observed_semantic_families": _counter_summary_limited(semantic_family_counter),
        "top_observed_semantic_signals": _counter_summary_limited(semantic_signal_counter),
    }


def _feedback_metrics(
    rows: list[dict],
    known_bug_families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float]:
    summary = aggregate_feedback_summary(rows, known_saturated_bug_families=known_bug_families)
    total_cases = max(1, len(rows))
    feedback_mutation_cases = int(summary["feedback_mutation_cases"])
    quality_oracle_count = int(summary["quality_oracle_count"])
    selected_operator_count = int(summary["feedback_selected_operator_count"])
    return {
        **summary,
        "feedback_mutation_case_rate": feedback_mutation_cases / total_cases,
        "feedback_corpus_store_rate": int(summary["stored_in_feedback_corpus_cases"]) / total_cases,
        "quality_pass_rate": (
            int(summary["quality_pass_count"]) / quality_oracle_count if quality_oracle_count else 0.0
        ),
        "quality_score_per_case": float(summary["quality_score_total"]) / total_cases,
        "source_reward_adjustment_per_case": float(summary["source_reward_adjustment_total"]) / total_cases,
        "guidance_reward_adjustment_per_case": float(summary["guidance_reward_adjustment_total"]) / total_cases,
        "seed_schedule_delta_per_case": float(summary["seed_schedule_delta_total"]) / total_cases,
        "productive_mutation_rate": (
            int(summary["productive_mutation_cases"]) / feedback_mutation_cases if feedback_mutation_cases else 0.0
        ),
        "invalid_mutation_rate": (
            int(summary["invalid_mutation_cases"]) / feedback_mutation_cases if feedback_mutation_cases else 0.0
        ),
        "guided_productive_rate": int(summary["guided_productive_cases"]) / total_cases,
        "guided_target_miss_rate": int(summary["guided_target_miss_cases"]) / total_cases,
        "feedback_operator_affinity_hit_rate": int(summary["feedback_operator_affinity_hit_cases"]) / total_cases,
        "feedback_selected_operator_score_avg": (
            float(summary["feedback_selected_operator_score_total"]) / selected_operator_count
            if selected_operator_count
            else 0.0
        ),
    }


def _avg_metric(rows: list[dict], key: str) -> float:
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _candidate_rate(rows: list[dict], numerator_key: str) -> float:
    numerator = sum(float(row.get(numerator_key, 0.0) or 0.0) for row in rows)
    denominator = sum(float(row.get("candidate_count", 0.0) or 0.0) for row in rows)
    return numerator / denominator if denominator else 0.0


def _common_capabilities_from_specs(target_specs: list[dict]) -> list[str]:
    if not target_specs:
        return []
    capabilities = [set(target.get("capabilities", [])) for target in target_specs]
    common = capabilities[0]
    for item in capabilities[1:]:
        common &= item
    return sorted(common)
