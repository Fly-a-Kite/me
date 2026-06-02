from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datadiff.experiment_catalog import FINAL_MODULE_ABLATION_MATRIX, resolve_final_experiment_variant
from datadiff.experiment_metadata import (
    component_focus,
    experiment_row_variant_id,
    experiment_row_variant_key,
    experiment_row_variant_label,
    is_ablation_contrast,
    is_contrast_variant,
    is_reference_variant,
    is_seeded_experiment_row,
    manifest_experiment_meta,
    reference_row_for_group,
    row_tag_set,
)
from datadiff.reporter import latest_experiment_manifest_path, write_experiment_summary_report
from datadiff.reward import (
    OFFLINE_BUCKET_FALSE_POSITIVE,
    OFFLINE_BUCKET_KNOWN_BUG,
    OFFLINE_BUCKET_NEEDS_TRIAGE,
    OFFLINE_BUCKET_NEW_BUG,
    OFFLINE_BUCKET_SEMANTIC_DIVERGENCE,
    OFFLINE_BUCKET_UNCLASSIFIED,
    candidate_issue_family_keys,
    offline_finding_buckets,
)
from datadiff.util import PROJECT_ROOT, REPORTS_DIR, dump_json, ensure_dirs, load_json, run_meta_path

OFFLINE_ORACLE_BUCKETS = (
    OFFLINE_BUCKET_NEW_BUG,
    OFFLINE_BUCKET_KNOWN_BUG,
    OFFLINE_BUCKET_FALSE_POSITIVE,
    OFFLINE_BUCKET_SEMANTIC_DIVERGENCE,
    OFFLINE_BUCKET_NEEDS_TRIAGE,
    OFFLINE_BUCKET_UNCLASSIFIED,
)

RUN_PROVENANCE_REQUIRED_FREEZE_ARTIFACTS = (
    "manifest",
    "pip_freeze",
    "git_status",
    "launcher_env",
    "strategy_snapshot",
)
RUN_PROVENANCE_OPTIONAL_FREEZE_ARTIFACTS = ("git_diff", "strategy_learning")
RUN_PROVENANCE_ARTIFACT_FIELDS = (
    *RUN_PROVENANCE_REQUIRED_FREEZE_ARTIFACTS,
    *RUN_PROVENANCE_OPTIONAL_FREEZE_ARTIFACTS,
)

latest_experiment_manifest = latest_experiment_manifest_path
write_experiment_summary = write_experiment_summary_report



def write_methodology_report(
    manifest_file: Path | None = None,
    *,
    refresh: bool = False,
    scan_run_logs: bool = True,
) -> tuple[Path, Path]:
    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_file or latest_experiment_manifest_path()
    summary_md, run_csv = _experiment_summary_paths(manifest_file)
    aggregate_csv = summary_md.with_name(f"{summary_md.stem}-aggregates.csv")
    aggregate_json = summary_md.with_name(f"{summary_md.stem}-aggregates.json")
    if refresh or not (summary_md.is_file() and run_csv.is_file() and aggregate_csv.is_file()):
        summary_md, run_csv = write_experiment_summary(manifest_file, refresh=refresh)
        aggregate_csv = summary_md.with_name(f"{summary_md.stem}-aggregates.csv")
        aggregate_json = summary_md.with_name(f"{summary_md.stem}-aggregates.json")
    run_rows = _load_csv_rows(run_csv)
    aggregate_rows = _load_structured_aggregate_rows(aggregate_json, aggregate_csv)
    manifest = load_json(manifest_file)
    report = _build_methodology_report(
        manifest_file,
        manifest,
        run_rows,
        aggregate_rows,
        scan_run_logs=scan_run_logs,
    )
    report["evidence_chain"] = {
        "manifest": str(manifest_file),
        "experiment_summary_markdown": str(summary_md),
        "run_csv": str(run_csv),
        "aggregate_csv": str(aggregate_csv),
        "aggregate_json": str(aggregate_json),
        "run_logs": [row.get("run_file", "") for row in run_rows if row.get("run_file")],
        "run_provenance_freeze_manifests": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_manifest_paths", []),
        "run_provenance_pip_freeze_artifacts": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_pip_freeze_paths", []),
        "run_provenance_git_status_artifacts": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_git_status_paths", []),
        "run_provenance_git_diff_artifacts": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_git_diff_paths", []),
        "run_provenance_launcher_env_artifacts": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_launcher_env_paths", []),
        "run_provenance_strategy_snapshot_artifacts": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_strategy_snapshot_paths", []),
        "run_provenance_strategy_learning_artifacts": report.get("reproducibility", {})
        .get("run_provenance", {})
        .get("freeze_strategy_learning_paths", []),
        "artifact_dirs": report.get("reproducibility", {}).get("artifact_dirs", []),
        "issue_bundle_manifest": report.get("reproducibility", {}).get("issue_bundle", {}).get("path", ""),
        "issue_bundle_reproducers": report.get("reproducibility", {})
        .get("issue_bundle", {})
        .get("reproducer_paths", []),
        "bug_sprint_manifests": report.get("scheduler_effectiveness", {}).get("manifest_paths", []),
        "candidate_pipeline_manifests": report.get("candidate_pipeline", {}).get("manifest_paths", []),
    }

    md_path = REPORTS_DIR / f"methodology-report-{manifest_file.stem}.md"
    json_path = REPORTS_DIR / f"methodology-report-{manifest_file.stem}.json"
    md_path.write_text(
        _render_markdown(report, summary_md, run_csv, aggregate_csv, aggregate_json),
        encoding="utf-8",
    )
    dump_json(report, json_path)
    return md_path, json_path


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_structured_aggregate_rows(
    aggregate_json_path: Path,
    aggregate_csv_path: Path,
) -> list[dict[str, Any]]:
    if aggregate_json_path.is_file():
        payload = json.loads(aggregate_json_path.read_text(encoding="utf-8"))
        variant_rows = payload.get("variant_rows", []) if isinstance(payload, dict) else []
        if isinstance(variant_rows, list) and all(isinstance(item, dict) for item in variant_rows):
            return variant_rows
    return _load_csv_rows(aggregate_csv_path)


def _experiment_summary_paths(manifest_file: Path) -> tuple[Path, Path]:
    summary_md = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}.md"
    run_csv = REPORTS_DIR / f"experiment-summary-{manifest_file.stem}.csv"
    return summary_md, run_csv


def _build_methodology_report(
    manifest_file: Path,
    manifest: dict[str, Any],
    run_rows: list[dict[str, str]],
    aggregate_rows: list[dict[str, str]],
    *,
    scan_run_logs: bool = True,
) -> dict[str, Any]:
    total_cases = sum(_int(row.get("cases")) for row in run_rows)
    total_elapsed_s = sum(_float(row.get("elapsed_s")) for row in run_rows)
    total_run_log_bytes = sum(
        _int_or_path_size(row.get("run_log_bytes"), row.get("run_file", "")) for row in run_rows
    )
    total_artifact_bytes = sum(_int(row.get("artifact_bytes")) for row in run_rows)
    total_evidence_bytes = sum(
        _int(row.get("evidence_bytes"))
        or (_int_or_path_size(row.get("run_log_bytes"), row.get("run_file", "")) + _int(row.get("artifact_bytes")))
        for row in run_rows
    )
    candidate_cases = sum(_int(row.get("candidate_bug_cases")) for row in run_rows)
    findings = sum(_int(row.get("findings")) for row in run_rows)
    false_positive_count = sum(
        _int(row.get("generator_false_positive_count")) + _int(row.get("normalizer_false_positive_count"))
        for row in run_rows
    )
    semantic_divergence_count = sum(
        _int(row.get("documented_semantic_divergence_count"))
        + _int(row.get("expected_semantic_divergence_count"))
        + _int(row.get("semantic_divergence_needs_confirmation_count"))
        for row in run_rows
    )
    families = Counter()
    for row in aggregate_rows:
        families.update(_parse_counter_summary(row.get("top_candidate_bug_families", "")))
    target_suites = sorted({row.get("target_suite", "") for row in aggregate_rows if row.get("target_suite")})
    presets = sorted({row.get("preset", "") for row in aggregate_rows if row.get("preset")})
    matrix_ids = sorted({row.get("matrix_id", "") for row in aggregate_rows if row.get("matrix_id")})
    comparison_groups = sorted({row.get("comparison_group", "") for row in aggregate_rows if row.get("comparison_group")})
    variant_ids = sorted({row.get("variant_id", "") for row in aggregate_rows if row.get("variant_id")})
    scope_kinds = sorted({row.get("scope_kind", "") for row in aggregate_rows if row.get("scope_kind")})
    oracle_profiles = sorted({row.get("oracle_profile", "") for row in aggregate_rows if row.get("oracle_profile")})
    rq_tags = sorted(_csv_union(aggregate_rows, "rq_tags"))
    analysis_tags = sorted(_csv_union(aggregate_rows, "analysis_tags"))
    semantic_focus_families = sorted(_csv_union(aggregate_rows, "semantic_focus_families"))
    semantic_focus_signals = sorted(_csv_union(aggregate_rows, "semantic_focus_signals"))
    evidence_modes = Counter(row.get("evidence_mode", "unknown") or "unknown" for row in run_rows)
    seeded_rows = [row for row in aggregate_rows if is_seeded_experiment_row(row)]
    ablation_rows = [row for row in aggregate_rows if _is_ablation_row(row)]
    reference_rows = [row for row in aggregate_rows if _is_reference_row(row)]
    expected_ablation_modules = _expected_ablation_modules(manifest, aggregate_rows)
    run_log_evidence = _run_log_evidence(run_rows) if scan_run_logs else _empty_run_log_evidence(run_rows)
    artifact_reproducibility = run_log_evidence["artifact_reproducibility"]
    offline_oracle = run_log_evidence["offline_oracle"]
    candidate_family_first_seen = run_log_evidence["candidate_family_first_seen"]
    run_provenance_reproducibility = _run_provenance_reproducibility(run_rows)
    issue_bundle_reproducibility = _issue_bundle_reproducibility(manifest_file)
    workflow_generated_dir = _workflow_generated_dir(manifest_file)
    scheduler_effectiveness = _scheduler_effectiveness(workflow_generated_dir)
    scheduler_effectiveness.update(_adaptive_scheduler_summary(manifest))
    candidate_pipeline = _candidate_pipeline_metrics(workflow_generated_dir)
    closed_loop = {
        "raw_new_behavior_cases": sum(_int(row.get("new_behavior_cases")) for row in run_rows),
        "signal_new_behavior_cases": sum(_int(row.get("signal_new_behavior_cases")) for row in run_rows),
        "feedback_case_count": sum(_int(row.get("feedback_case_count")) for row in run_rows),
        "feedback_mutation_cases": sum(_int(row.get("feedback_mutation_cases")) for row in run_rows),
        "feedback_target_key_count": sum(_int(row.get("feedback_target_key_count")) for row in run_rows),
        "feedback_semantic_family_target_count": sum(
            _int(row.get("feedback_semantic_family_target_count")) for row in run_rows
        ),
        "feedback_semantic_signal_target_count": sum(
            _int(row.get("feedback_semantic_signal_target_count")) for row in run_rows
        ),
        "feedback_operator_affinity_hit_cases": sum(
            _int(row.get("feedback_operator_affinity_hit_cases")) for row in run_rows
        ),
        "feedback_selected_operator_count": sum(_int(row.get("feedback_selected_operator_count")) for row in run_rows),
        "stored_in_feedback_corpus_cases": sum(_int(row.get("stored_in_feedback_corpus_cases")) for row in run_rows),
        "quality_oracle_count": sum(_int(row.get("quality_oracle_count")) for row in run_rows),
        "quality_pass_count": sum(_int(row.get("quality_pass_count")) for row in run_rows),
        "quality_fail_count": sum(_int(row.get("quality_fail_count")) for row in run_rows),
        "quality_score_total": sum(_float(row.get("quality_score_total")) for row in run_rows),
        "feedback_selected_operator_score_total": sum(
            _float(row.get("feedback_selected_operator_score_total")) for row in run_rows
        ),
        "source_reward_adjustment_total": sum(_float(row.get("source_reward_adjustment_total")) for row in run_rows),
        "guidance_reward_adjustment_total": sum(_float(row.get("guidance_reward_adjustment_total")) for row in run_rows),
        "seed_schedule_delta_total": sum(_float(row.get("seed_schedule_delta_total")) for row in run_rows),
        "productive_mutation_cases": sum(_int(row.get("productive_mutation_cases")) for row in run_rows),
        "invalid_mutation_cases": sum(_int(row.get("invalid_mutation_cases")) for row in run_rows),
        "feedback_finding_yield_cases": sum(_int(row.get("feedback_finding_yield_cases")) for row in run_rows),
        "feedback_new_behavior_yield_cases": sum(_int(row.get("feedback_new_behavior_yield_cases")) for row in run_rows),
        "feedback_redundant_behavior_cases": sum(_int(row.get("feedback_redundant_behavior_cases")) for row in run_rows),
        "guided_productive_cases": sum(_int(row.get("guided_productive_cases")) for row in run_rows),
        "guided_target_miss_cases": sum(_int(row.get("guided_target_miss_cases")) for row in run_rows),
        "raw_new_behavior_rate": (
            sum(_int(row.get("new_behavior_cases")) for row in run_rows) / total_cases if total_cases else 0.0
        ),
        "signal_new_behavior_rate": (
            sum(_int(row.get("signal_new_behavior_cases")) for row in run_rows) / total_cases if total_cases else 0.0
        ),
        "feedback_mutation_case_rate": _avg_float(aggregate_rows, "feedback_mutation_case_rate"),
        "feedback_operator_affinity_hit_rate": _avg_float(aggregate_rows, "feedback_operator_affinity_hit_rate"),
        "feedback_selected_operator_score_avg": _avg_float(aggregate_rows, "feedback_selected_operator_score_avg"),
        "feedback_corpus_store_rate": _avg_float(aggregate_rows, "feedback_corpus_store_rate"),
        "quality_pass_rate": _avg_float(aggregate_rows, "quality_pass_rate"),
        "quality_score_per_case": _avg_float(aggregate_rows, "quality_score_per_case"),
        "source_reward_adjustment_per_case": _avg_float(aggregate_rows, "source_reward_adjustment_per_case"),
        "guidance_reward_adjustment_per_case": _avg_float(aggregate_rows, "guidance_reward_adjustment_per_case"),
        "seed_schedule_delta_per_case": _avg_float(aggregate_rows, "seed_schedule_delta_per_case"),
        "productive_mutation_rate": _avg_float(aggregate_rows, "productive_mutation_rate"),
        "guided_productive_rate": _avg_float(aggregate_rows, "guided_productive_rate"),
        "guided_target_miss_rate": _avg_float(aggregate_rows, "guided_target_miss_rate"),
        "semantic_focus_families": semantic_focus_families,
        "semantic_focus_signals": semantic_focus_signals,
        "configured_guidance_targets": sorted(_csv_union(aggregate_rows, "configured_guidance_targets")),
        "configured_effective_guidance_targets": sorted(
            _csv_union(aggregate_rows, "configured_effective_guidance_targets")
        ),
        "configured_semantic_focus_families": sorted(_csv_union(aggregate_rows, "configured_semantic_focus_families")),
        "configured_semantic_focus_signals": sorted(_csv_union(aggregate_rows, "configured_semantic_focus_signals")),
        "configured_discovery_biases": sorted(_csv_union(aggregate_rows, "configured_discovery_biases", delimiter=";")),
        "matched_semantic_target_cases": sum(_int(row.get("matched_semantic_target_cases")) for row in aggregate_rows),
        "discovery_bias_hit_cases": sum(_int(row.get("discovery_bias_hit_cases")) for row in aggregate_rows),
        "matched_semantic_target_case_rate": _avg_float(aggregate_rows, "matched_semantic_target_case_rate"),
        "discovery_bias_hit_case_rate": _avg_float(aggregate_rows, "discovery_bias_hit_case_rate"),
        "avg_matched_semantic_target_count": _avg_float(aggregate_rows, "avg_matched_semantic_target_count"),
        "avg_discovery_bias_hit_count": _avg_float(aggregate_rows, "avg_discovery_bias_hit_count"),
        "top_matched_semantic_targets": _merge_counter_summaries(aggregate_rows, "top_matched_semantic_targets"),
        "top_discovery_bias_hits": _merge_counter_summaries(aggregate_rows, "top_discovery_bias_hits"),
        "top_observed_semantic_families": _merge_counter_summaries(aggregate_rows, "top_observed_semantic_families"),
        "top_observed_semantic_signals": _merge_counter_summaries(aggregate_rows, "top_observed_semantic_signals"),
        "top_feedback_selected_operators": _merge_counter_summaries(aggregate_rows, "top_feedback_selected_operators"),
        "top_feedback_semantic_target_keys": _merge_counter_summaries(
            aggregate_rows,
            "top_feedback_semantic_target_keys",
        ),
    }
    stage_profile = {
        "generate_mutate_total_ms": sum(_float(row.get("stage_generate_mutate_total_ms")) for row in aggregate_rows),
        "backend_execution_total_ms": sum(_float(row.get("stage_backend_execution_total_ms")) for row in aggregate_rows),
        "normalize_total_ms": sum(_float(row.get("stage_normalize_total_ms")) for row in aggregate_rows),
        "oracle_classification_total_ms": sum(
            _float(row.get("stage_oracle_classification_total_ms")) for row in aggregate_rows
        ),
        "scheduler_feedback_total_ms": sum(
            _float(row.get("stage_scheduler_feedback_total_ms")) for row in aggregate_rows
        ),
        "logging_artifact_total_ms": sum(_float(row.get("stage_logging_artifact_total_ms")) for row in aggregate_rows),
        "total_case_wall_ms": sum(_float(row.get("stage_total_case_wall_ms")) for row in aggregate_rows),
        "generate_mutate_avg_ms": _avg_float(aggregate_rows, "stage_generate_mutate_avg_ms"),
        "backend_execution_avg_ms": _avg_float(aggregate_rows, "stage_backend_execution_avg_ms"),
        "normalize_avg_ms": _avg_float(aggregate_rows, "stage_normalize_avg_ms"),
        "oracle_classification_avg_ms": _avg_float(aggregate_rows, "stage_oracle_classification_avg_ms"),
        "scheduler_feedback_avg_ms": _avg_float(aggregate_rows, "stage_scheduler_feedback_avg_ms"),
        "logging_artifact_avg_ms": _avg_float(aggregate_rows, "stage_logging_artifact_avg_ms"),
        "total_case_wall_avg_ms": _avg_float(aggregate_rows, "stage_total_case_wall_avg_ms"),
        "backend_execution_share": _avg_float(aggregate_rows, "stage_backend_execution_share"),
        "oracle_classification_share": _avg_float(aggregate_rows, "stage_oracle_classification_share"),
        "scheduler_feedback_share": _avg_float(aggregate_rows, "stage_scheduler_feedback_share"),
    }

    return {
        "schema_version": "methodology-report-v1",
        "manifest_file": str(manifest_file),
        "created_from": {
            "experiment_schedule": manifest.get("schedule", "matrix_order"),
            "target_suites": manifest.get("target_suites", target_suites),
            "presets": manifest.get("presets", presets),
            "seeds": manifest.get("seeds", []),
            "evidence_mode": manifest.get("evidence_mode", ""),
        },
        "workflow": [
            "integrate_novel_exploration_mode",
            "short_validation_then_long_discovery",
            "multi_module_ablation_and_comparison",
            "efficiency_discovery_reproducibility_audit",
        ],
        "coverage": {
            "target_suite_count": len(target_suites),
            "target_suites": target_suites,
            "preset_count": len(presets),
            "presets": presets,
            "matrix_ids": matrix_ids,
            "comparison_groups": comparison_groups,
            "variant_count": len(variant_ids),
            "variant_ids": variant_ids,
            "scope_kinds": scope_kinds,
            "oracle_profiles": oracle_profiles,
            "rq_tags": rq_tags,
            "analysis_tags": analysis_tags,
            "semantic_focus_families": semantic_focus_families,
            "semantic_focus_signals": semantic_focus_signals,
            "run_count": len(run_rows),
            "evidence_modes": dict(evidence_modes.most_common()),
        },
        "efficiency": {
            "cases": total_cases,
            "elapsed_s": total_elapsed_s,
            "cases_per_s": total_cases / total_elapsed_s if total_elapsed_s > 0 else 0.0,
            "run_log_bytes": total_run_log_bytes,
            "artifact_bytes": total_artifact_bytes,
            "evidence_bytes": total_evidence_bytes,
            "run_log_bytes_per_case": total_run_log_bytes / total_cases if total_cases else 0.0,
            "artifact_bytes_per_case": total_artifact_bytes / total_cases if total_cases else 0.0,
            "evidence_bytes_per_case": total_evidence_bytes / total_cases if total_cases else 0.0,
            "stage_profile": stage_profile,
        },
        "bug_discovery": {
            "findings": findings,
            "candidate_bug_cases": candidate_cases,
            "candidate_bug_case_rate": candidate_cases / total_cases if total_cases else 0.0,
            "candidate_bug_families": dict(families.most_common()),
            "candidate_bug_family_count": len(families),
            "first_candidate": _global_first_candidate(run_rows),
            "candidate_family_first_seen": candidate_family_first_seen,
            "candidate_family_first_seen_count": len(candidate_family_first_seen),
            "avg_candidate_bug_discovery_auc": _avg_float(aggregate_rows, "avg_candidate_bug_discovery_auc"),
        },
        "soundness": {
            "semantic_divergence_count": semantic_divergence_count,
            "false_positive_count": false_positive_count,
            "false_positive_rate": false_positive_count / findings if findings else 0.0,
            "preflight_invalid_cases": sum(_int(row.get("preflight_invalid_cases")) for row in aggregate_rows),
            "known_saturated_candidate_count": sum(
                _int(row.get("known_saturated_candidate_bug_count")) for row in aggregate_rows
            ),
        },
        "closed_loop_feedback": closed_loop,
        "scheduler_effectiveness": scheduler_effectiveness,
        "candidate_pipeline": candidate_pipeline,
        "offline_oracle": offline_oracle,
        "run_log_scan": run_log_evidence["run_log_scan"],
        "reproducibility": {
            "seeded_suite_count": len(seeded_rows),
            "seeded_cases": sum(_int(row.get("cases")) for row in seeded_rows),
            "seeded_candidate_bug_cases": sum(_int(row.get("candidate_bug_cases")) for row in seeded_rows),
            "has_seeded_sensitivity_axis": bool(seeded_rows),
            "run_logs_exist": sum(1 for row in run_rows if _resolve_existing_path(row.get("run_file", ""))),
            "run_logs_total": len(run_rows),
            **artifact_reproducibility,
            "run_provenance": run_provenance_reproducibility,
            "issue_bundle": issue_bundle_reproducibility,
        },
        "ablation": {
            "reference_run_groups": len(reference_rows),
            "baseline_run_groups": len(reference_rows),
            "ablation_run_groups": len(ablation_rows),
            "ablation_presets": sorted({row.get("preset", "") for row in ablation_rows if row.get("preset")}),
            "ablation_variant_ids": sorted({row.get("variant_id", "") for row in ablation_rows if row.get("variant_id")}),
            "ablation_modules": sorted(
                {
                    component_focus(row)
                    for row in ablation_rows
                    if component_focus(row)
                }
            ),
            "expected_ablation_modules": expected_ablation_modules,
            "missing_ablation_modules": sorted(
                set(expected_ablation_modules)
                - {
                    component_focus(row)
                    for row in ablation_rows
                    if component_focus(row)
                }
            ),
            "ablation_false_positive_count": sum(_int(row.get("false_positive_count")) for row in ablation_rows),
            "ablation_candidate_bug_cases": sum(_int(row.get("candidate_bug_cases")) for row in ablation_rows),
        },
        "comparisons": _reference_variant_comparisons(aggregate_rows),
    }


def _reference_variant_comparisons(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    structured_contrast_keys = {
        experiment_row_variant_key(row)
        for row in rows
        if is_contrast_variant(row)
    }
    candidate_rows = [
        row
        for row in rows
        if not _is_reference_row(row)
        and (
            not structured_contrast_keys
            or experiment_row_variant_key(row) in structured_contrast_keys
        )
    ]
    for row in candidate_rows:
        suite = row.get("target_suite", "")
        comparison_variant_label = experiment_row_variant_label(row)
        if not suite:
            continue
        reference_row = _reference_row_for_group(rows, row)
        if reference_row is None:
            continue
        current_rate = _float(row.get("candidate_bug_case_rate"))
        baseline_rate = _float(reference_row.get("candidate_bug_case_rate"))
        current_throughput = _float(row.get("avg_throughput_cases_s"))
        baseline_throughput = _float(reference_row.get("avg_throughput_cases_s"))
        current_evidence_bytes = _float(row.get("evidence_bytes_per_case"))
        baseline_evidence_bytes = _float(reference_row.get("evidence_bytes_per_case"))
        comparisons.append(
            {
                "target_suite": suite,
                "preset": row.get("preset", ""),
                "variant_label": comparison_variant_label,
                "matrix_id": row.get("matrix_id", ""),
                "comparison_group": row.get("comparison_group", ""),
                "variant_id": row.get("variant_id", ""),
                "reference_variant_id": experiment_row_variant_id(reference_row),
                "reference_variant_label": experiment_row_variant_label(reference_row),
                "candidate_bug_case_rate_delta": current_rate - baseline_rate,
                "candidate_bug_case_rate_ratio": _ratio(current_rate, baseline_rate),
                "throughput_cases_s_delta": current_throughput - baseline_throughput,
                "throughput_cases_s_ratio": _ratio(current_throughput, baseline_throughput),
                "evidence_bytes_per_case_delta": current_evidence_bytes - baseline_evidence_bytes,
                "evidence_bytes_per_case_ratio": _ratio(current_evidence_bytes, baseline_evidence_bytes),
                "false_positive_delta": _int(row.get("false_positive_count"))
                - _int(reference_row.get("false_positive_count")),
            }
        )
    return comparisons


_reference_comparisons = _reference_variant_comparisons


def _csv_union(rows: list[dict[str, str]], key: str, *, delimiter: str = ",") -> set[str]:
    values: set[str] = set()
    for row in rows:
        for item in str(row.get(key, "") or "").split(delimiter):
            text = item.strip()
            if text:
                values.add(text)
    return values


def _merge_counter_summaries(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(_parse_counter_summary(str(row.get(key, "") or "")))
    return dict(counter.most_common())


def _tag_set(row: dict[str, str], key: str = "analysis_tags") -> set[str]:
    return row_tag_set(row, key)


def _is_reference_row(row: dict[str, str]) -> bool:
    return is_reference_variant(row)


def _is_reference_row_compat(row: dict[str, str]) -> bool:
    return _is_reference_row(row)


def _is_ablation_row(row: dict[str, str]) -> bool:
    return is_ablation_contrast(row)


_is_baseline_row = _is_reference_row_compat


def _expected_ablation_modules(manifest: dict[str, Any], rows: list[dict[str, str]]) -> list[str]:
    matrix_ids = {
        str(row.get("matrix_id", "") or "").strip()
        for row in rows
        if str(row.get("matrix_id", "") or "").strip()
    }
    experiment_meta = manifest_experiment_meta(manifest)
    variant_by_preset = experiment_meta.get("variant_by_preset", {})
    manifest_presets = {
        str(preset).strip()
        for preset in manifest.get("presets", []) or []
        if str(preset).strip()
    }
    final_ablation_presets = {variant.preset for variant in FINAL_MODULE_ABLATION_MATRIX.variants}
    structured_final_ablation = (
        "module_ablation" in matrix_ids
        and isinstance(variant_by_preset, dict)
        and final_ablation_presets.issubset({str(key).strip() for key in variant_by_preset})
        and bool(manifest_presets & final_ablation_presets)
    )
    if structured_final_ablation:
        return sorted(
            {
                variant.component_focus
                for variant in FINAL_MODULE_ABLATION_MATRIX.variants
                if variant.component_focus
            }
        )
    registered_modules = {
        variant.component_focus
        for row in rows
        if (
            variant := resolve_final_experiment_variant(
                str(row.get("matrix_id", "") or ""),
                preset=str(row.get("preset", "") or ""),
                variant_id=str(row.get("variant_id", "") or ""),
            )
        )
        is not None
        and variant.component_focus
    }
    if registered_modules:
        return sorted(registered_modules)
    return sorted(
        {
            component_focus(row)
            for row in rows
            if "ablation" in _tag_set(row) and component_focus(row)
        }
    )


def _reference_row_for_group(
    rows: list[dict[str, str]],
    row: dict[str, str],
) -> dict[str, str] | None:
    return reference_row_for_group(rows, row)


def _render_markdown(
    report: dict[str, Any],
    summary_md: Path,
    run_csv: Path,
    aggregate_csv: Path,
    aggregate_json: Path,
) -> str:
    efficiency = report["efficiency"]
    discovery = report["bug_discovery"]
    soundness = report["soundness"]
    scheduler_effectiveness = report.get("scheduler_effectiveness", {})
    candidate_pipeline = report.get("candidate_pipeline", {})
    run_log_scan = report.get("run_log_scan", {})
    reproducibility = report["reproducibility"]
    run_provenance = reproducibility.get("run_provenance", {})
    issue_bundle = reproducibility.get("issue_bundle", {})
    ablation = report["ablation"]
    offline_oracle = report["offline_oracle"]
    closed_loop = report.get("closed_loop_feedback", {})
    evidence_chain = report.get("evidence_chain", {})
    lines = [
        "# DataDiffFuzz Methodology Report",
        "",
        f"- Manifest: `{report['manifest_file']}`",
        f"- Experiment summary: `{summary_md}`",
        f"- Run CSV: `{run_csv}`",
        f"- Aggregate CSV: `{aggregate_csv}`",
        f"- Aggregate JSON: `{aggregate_json}`",
        "",
        "## Methodology Loop",
        "",
        "1. Integrate a novel exploration mode into generator, scheduler, adapter/oracle, classification, and evidence layers.",
        "2. Run short validation to catch adapter and oracle noise, then run longer discovery with health gates.",
            "3. Compare modules by ablation, reference/contrast variants, seeded replay sensitivity, and efficiency metrics.",
        "",
        "## Coverage",
        "",
        f"- Target suites: {report['coverage']['target_suite_count']} ({', '.join(report['coverage']['target_suites'])})",
        f"- Presets executed: {report['coverage']['preset_count']} ({', '.join(report['coverage']['presets'])})",
        f"- Matrix ids: {len(report['coverage']['matrix_ids'])} ({', '.join(report['coverage']['matrix_ids']) or 'none'})",
        f"- Comparison groups: {len(report['coverage']['comparison_groups'])} ({', '.join(report['coverage']['comparison_groups']) or 'none'})",
        f"- Variants: {report['coverage']['variant_count']} ({', '.join(report['coverage']['variant_ids']) or 'none'})",
        f"- Scope kinds: {', '.join(report['coverage']['scope_kinds']) or 'none'}",
        f"- Oracle profiles: {', '.join(report['coverage']['oracle_profiles']) or 'none'}",
        f"- RQ tags: {', '.join(report['coverage']['rq_tags']) or 'none'}",
        f"- Analysis tags: {', '.join(report['coverage']['analysis_tags']) or 'none'}",
        f"- Structured semantic focus families: {', '.join(report['coverage']['semantic_focus_families']) or 'none'}",
        f"- Structured semantic focus signals: {', '.join(report['coverage']['semantic_focus_signals']) or 'none'}",
        f"- Runs: {report['coverage']['run_count']}",
        f"- Evidence modes: {_counter_text(report['coverage']['evidence_modes'])}",
        "",
        "## Efficiency",
        "",
        f"- Cases: {efficiency['cases']}",
        f"- Elapsed seconds: {_fmt_float(efficiency['elapsed_s'])}",
        f"- Cases/s: {_fmt_float(efficiency['cases_per_s'])}",
        f"- Run log bytes: {efficiency['run_log_bytes']}",
        f"- Artifact bytes: {efficiency['artifact_bytes']}",
        f"- Evidence bytes: {efficiency['evidence_bytes']}",
        f"- Evidence bytes/case: {_fmt_float(efficiency['evidence_bytes_per_case'])}",
        f"- Stage gen/mutate avg ms: {_fmt_float(efficiency['stage_profile']['generate_mutate_avg_ms'])}",
        f"- Stage backend avg ms: {_fmt_float(efficiency['stage_profile']['backend_execution_avg_ms'])} ({_fmt_percent(efficiency['stage_profile']['backend_execution_share'])})",
        f"- Stage normalize avg ms: {_fmt_float(efficiency['stage_profile']['normalize_avg_ms'])}",
        f"- Stage oracle avg ms: {_fmt_float(efficiency['stage_profile']['oracle_classification_avg_ms'])} ({_fmt_percent(efficiency['stage_profile']['oracle_classification_share'])})",
        f"- Stage scheduler avg ms: {_fmt_float(efficiency['stage_profile']['scheduler_feedback_avg_ms'])} ({_fmt_percent(efficiency['stage_profile']['scheduler_feedback_share'])})",
        f"- Stage logging avg ms: {_fmt_float(efficiency['stage_profile']['logging_artifact_avg_ms'])}",
        "",
        "## Bug Discovery",
        "",
        f"- Findings: {discovery['findings']}",
        f"- Candidate bug cases: {discovery['candidate_bug_cases']} ({_fmt_percent(discovery['candidate_bug_case_rate'])})",
        f"- Candidate bug families: {discovery['candidate_bug_family_count']}",
        f"- First candidate: {discovery['first_candidate'] or 'none'}",
        f"- Candidate family first-seen records: {discovery['candidate_family_first_seen_count']}",
        f"- Avg discovery AUC: {_fmt_float(discovery['avg_candidate_bug_discovery_auc'])}",
        "",
        "### Candidate Family First Seen",
        "",
    ]
    if discovery["candidate_family_first_seen"]:
        lines.extend(
            [
                "| family | target suite | variant | preset | seed | case index | elapsed s |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for family, row in discovery["candidate_family_first_seen"].items():
            lines.append(
                "| {family} | {target_suite} | {variant_label} | {preset} | {seed} | {case_index} | {elapsed_s} |".format(
                    family=family,
                    target_suite=row.get("target_suite", ""),
                    variant_label=row.get("variant_label", ""),
                    preset=row.get("preset", ""),
                    seed=row.get("seed", ""),
                    case_index=row.get("case_index", ""),
                    elapsed_s=_fmt_optional_number(row.get("elapsed_s")),
                )
            )
        lines.append("")
    else:
        lines.extend(["No candidate family first-seen records.", ""])
    lines.extend(
        [
            "## Soundness",
            "",
            f"- Semantic divergence count: {soundness['semantic_divergence_count']}",
            f"- False positive count: {soundness['false_positive_count']} ({_fmt_percent(soundness['false_positive_rate'])} of findings)",
            f"- Preflight invalid cases: {soundness['preflight_invalid_cases']}",
            f"- Known saturated candidate count: {soundness['known_saturated_candidate_count']}",
            "",
            "## Closed-Loop Feedback",
            "",
            f"- Raw new behavior cases: {closed_loop.get('raw_new_behavior_cases', 0)} ({_fmt_percent(closed_loop.get('raw_new_behavior_rate', 0.0))})",
            f"- Signal new behavior cases: {closed_loop.get('signal_new_behavior_cases', 0)} ({_fmt_percent(closed_loop.get('signal_new_behavior_rate', 0.0))})",
            f"- Feedback mutation case rate: {_fmt_percent(closed_loop.get('feedback_mutation_case_rate', 0.0))}",
            f"- Feedback operator affinity-hit rate: {_fmt_percent(closed_loop.get('feedback_operator_affinity_hit_rate', 0.0))}",
            f"- Avg selected feedback-operator score: {_fmt_float(closed_loop.get('feedback_selected_operator_score_avg', 0.0))}",
            f"- Feedback corpus store rate: {_fmt_percent(closed_loop.get('feedback_corpus_store_rate', 0.0))}",
            f"- Quality pass rate: {_fmt_percent(closed_loop.get('quality_pass_rate', 0.0))}",
            f"- Productive mutation rate: {_fmt_percent(closed_loop.get('productive_mutation_rate', 0.0))}",
            f"- Guided productive rate: {_fmt_percent(closed_loop.get('guided_productive_rate', 0.0))}",
            f"- Guided target-miss rate: {_fmt_percent(closed_loop.get('guided_target_miss_rate', 0.0))}",
            f"- Structured semantic focus families: {', '.join(closed_loop.get('semantic_focus_families', [])) or 'none'}",
            f"- Structured semantic focus signals: {', '.join(closed_loop.get('semantic_focus_signals', [])) or 'none'}",
            f"- Configured guidance targets: {', '.join(closed_loop.get('configured_guidance_targets', [])) or 'none'}",
            f"- Effective guidance targets: {', '.join(closed_loop.get('configured_effective_guidance_targets', [])) or 'none'}",
            f"- Configured semantic focus families: {', '.join(closed_loop.get('configured_semantic_focus_families', [])) or 'none'}",
            f"- Configured semantic focus signals: {', '.join(closed_loop.get('configured_semantic_focus_signals', [])) or 'none'}",
            f"- Configured discovery biases: {', '.join(closed_loop.get('configured_discovery_biases', [])) or 'none'}",
            f"- Matched semantic target cases: {closed_loop.get('matched_semantic_target_cases', 0)} ({_fmt_percent(closed_loop.get('matched_semantic_target_case_rate', 0.0))})",
            f"- Discovery-bias hit cases: {closed_loop.get('discovery_bias_hit_cases', 0)} ({_fmt_percent(closed_loop.get('discovery_bias_hit_case_rate', 0.0))})",
            f"- Avg matched semantic targets/case: {_fmt_float(closed_loop.get('avg_matched_semantic_target_count', 0.0))}",
            f"- Avg discovery-bias hits/case: {_fmt_float(closed_loop.get('avg_discovery_bias_hit_count', 0.0))}",
            f"- Top matched semantic targets: {_counter_text(closed_loop.get('top_matched_semantic_targets', {}))}",
            f"- Top discovery-bias hits: {_counter_text(closed_loop.get('top_discovery_bias_hits', {}))}",
            f"- Top observed semantic families: {_counter_text(closed_loop.get('top_observed_semantic_families', {}))}",
            f"- Top observed semantic signals: {_counter_text(closed_loop.get('top_observed_semantic_signals', {}))}",
            f"- Top selected feedback operators: {_counter_text(closed_loop.get('top_feedback_selected_operators', {}))}",
            f"- Top feedback semantic target keys: {_counter_text(closed_loop.get('top_feedback_semantic_target_keys', {}))}",
            f"- Source reward adjustment per case: {_fmt_float(closed_loop.get('source_reward_adjustment_per_case', 0.0))}",
            f"- Guidance reward adjustment per case: {_fmt_float(closed_loop.get('guidance_reward_adjustment_per_case', 0.0))}",
            f"- Seed schedule delta per case: {_fmt_float(closed_loop.get('seed_schedule_delta_per_case', 0.0))}",
            "",
            "## Scheduler Effectiveness",
            "",
            f"- Bug-sprint manifests: {scheduler_effectiveness.get('manifest_count', 0)}",
            f"- Completed lane runs: {scheduler_effectiveness.get('completed_run_count', 0)}",
            f"- Lane score samples: {scheduler_effectiveness.get('lane_score_sample_count', 0)}",
            f"- Avg lane score: {_fmt_float(scheduler_effectiveness.get('avg_score', 0.0))}",
            f"- Avg budget multiplier: {_fmt_float(scheduler_effectiveness.get('avg_budget_multiplier', 0.0))}",
            f"- Adaptive schedule enabled: {str(scheduler_effectiveness.get('adaptive_enabled', False)).lower()}",
            f"- Adaptive final arms: {scheduler_effectiveness.get('adaptive_final_arm_count', 0)}",
            f"- Adaptive avg reward signal: {_fmt_float(scheduler_effectiveness.get('adaptive_avg_reward_signal', 0.0))}",
            f"- Adaptive max reward signal: {_fmt_float(scheduler_effectiveness.get('adaptive_max_reward_signal', 0.0))}",
            f"- Adaptive avg mean reward: {_fmt_float(scheduler_effectiveness.get('adaptive_avg_mean_reward', 0.0))}",
            f"- Local source scheduler: {scheduler_effectiveness.get('local_source_scheduler', {'enabled': False})}",
            "",
            "| lane | completed runs | fresh candidates | unique fresh families | avg score | avg budget | avg yield | avg novelty | avg fp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lane_rows = scheduler_effectiveness.get("lane_yield", [])
    if lane_rows:
        for lane in lane_rows:
            lines.append(
                "| {lane_id} | {completed_runs} | {fresh_candidate_total} | {unique_fresh_family_count} | "
                "{avg_score} | {avg_budget} | {avg_yield} | {avg_novelty} | {avg_fp} |".format(
                    lane_id=lane.get("lane_id", ""),
                    completed_runs=lane.get("completed_runs", 0),
                    fresh_candidate_total=lane.get("fresh_candidate_total", 0),
                    unique_fresh_family_count=lane.get("unique_fresh_family_count", 0),
                    avg_score=_fmt_float(lane.get("avg_score", 0.0)),
                    avg_budget=_fmt_float(lane.get("avg_budget_multiplier", 0.0)),
                    avg_yield=_fmt_float(lane.get("avg_yield_rate", 0.0)),
                    avg_novelty=_fmt_float(lane.get("avg_novelty_rate", 0.0)),
                    avg_fp=_fmt_float(lane.get("avg_false_positive_rate", 0.0)),
                )
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |")
    adaptive_arms = scheduler_effectiveness.get("adaptive_final_arms", [])
    lines.extend(
        [
            "",
            "### Adaptive Final Arms",
            "",
            "| arm | target suite | pulls | mean reward | reward signal | last reward | stale batches |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    if adaptive_arms:
        for arm in adaptive_arms:
            lines.append(
                "| {arm_id} | {target_suite} | {pulls} | {mean_reward} | {reward_signal} | {last_reward} | {stale_batches} |".format(
                    arm_id=arm.get("arm_id", ""),
                    target_suite=arm.get("target_suite", ""),
                    pulls=int(arm.get("pulls", 0) or 0),
                    mean_reward=_fmt_float(arm.get("mean_reward", 0.0)),
                    reward_signal=_fmt_float(arm.get("reward_signal", 0.0)),
                    last_reward=_fmt_float(arm.get("last_reward", 0.0)),
                    stale_batches=int(arm.get("stale_batches", 0) or 0),
                )
            )
    else:
        lines.append("| none | none | 0 | 0.00 | 0.00 | 0.00 | 0 |")
    lines.extend(
        [
            "",
            "## Candidate Pipeline",
            "",
            f"- Pipeline manifests: {candidate_pipeline.get('manifest_count', 0)}",
            f"- Frozen candidates: {candidate_pipeline.get('candidate_count', 0)}",
            f"- Rechecked candidates: {candidate_pipeline.get('rechecked_count', 0)}",
            f"- Reproduced candidates: {candidate_pipeline.get('reproduced_count', 0)} ({_fmt_percent(candidate_pipeline.get('recheck_pass_rate', 0.0))})",
            f"- Reduced artifacts: {candidate_pipeline.get('reduced_count', 0)}",
            f"- Candidate-bug triage verdicts: {candidate_pipeline.get('candidate_bug_verdict_count', 0)}",
            f"- Issue drafts: {candidate_pipeline.get('issue_draft_count', 0)}",
            f"- Needs dedup check: {candidate_pipeline.get('needs_dedup_check_count', 0)}",
            f"- Already confirmed/submitted: {candidate_pipeline.get('already_submitted_or_confirmed_count', 0)}",
            "",
            "## Offline Oracle Buckets",
            "",
            f"- Classified findings: {offline_oracle['classified_findings']}",
            f"- New bug: {offline_oracle['buckets'][OFFLINE_BUCKET_NEW_BUG]}",
            f"- Known bug: {offline_oracle['buckets'][OFFLINE_BUCKET_KNOWN_BUG]}",
            f"- False positive: {offline_oracle['buckets'][OFFLINE_BUCKET_FALSE_POSITIVE]}",
            f"- Semantic divergence: {offline_oracle['buckets'][OFFLINE_BUCKET_SEMANTIC_DIVERGENCE]}",
            f"- Needs triage: {offline_oracle['buckets'][OFFLINE_BUCKET_NEEDS_TRIAGE]}",
            f"- Unclassified: {offline_oracle['buckets'][OFFLINE_BUCKET_UNCLASSIFIED]}",
            f"- Run-log scan skipped: {str(run_log_scan.get('run_logs_scan_skipped', False)).lower()}",
            f"- Run logs scanned: {run_log_scan.get('run_logs_scanned', 0)}/{run_log_scan.get('run_logs_total', 0)}",
            "",
            "## Reproducibility",
            "",
            f"- Seeded sensitivity axis present: {str(reproducibility['has_seeded_sensitivity_axis']).lower()}",
            f"- Seeded cases: {reproducibility['seeded_cases']}",
            f"- Seeded candidate bug cases: {reproducibility['seeded_candidate_bug_cases']}",
            f"- Run logs available: {reproducibility['run_logs_exist']}/{reproducibility['run_logs_total']}",
            f"- Run provenance recorded: {run_provenance.get('run_count_with_provenance', 0)}/{run_provenance.get('run_logs_total', 0)}",
            f"- Authority runs: {run_provenance.get('authority_run_count', 0)}",
            f"- Freeze intent declared: {run_provenance.get('freeze_intent_run_count', 0)}",
            f"- Latest-code claimed: {run_provenance.get('latest_code_claim_run_count', 0)}",
            f"- Clean workspace provenance: {run_provenance.get('clean_workspace_run_count', 0)}",
            f"- Fully qualified latest-live authority runs: {run_provenance.get('qualified_live_authority_run_count', 0)}",
            f"- All live runs satisfy authority/freeze gates: {str(run_provenance.get('all_live_runs_have_qualified_authority_provenance', False)).lower()}",
            f"- Required freeze artifact coverage: {_counter_text(run_provenance.get('freeze_artifact_present_counts', {}))}",
            f"- Missing provenance labels: {', '.join(run_provenance.get('missing_labels', [])) or 'none'}",
            f"- Artifact dirs existing/referenced: {reproducibility['artifact_dirs_existing']}/{reproducibility['artifact_dirs_total']}",
            f"- Artifact reproducer coverage: {_fmt_percent(reproducibility['artifact_reproducer_coverage'])}",
            f"- Reduced reproducers: {reproducibility['artifact_dirs_with_reduced_reproducer']}",
            f"- Standalone reproducers: {reproducibility['artifact_dirs_with_standalone_reproducer']}",
            f"- Triage reports: {reproducibility['artifact_dirs_with_triage_report']}",
            f"- Issue bundle present: {str(issue_bundle.get('present', False)).lower()}",
            f"- Issue bundle families: {issue_bundle.get('family_count', 0)}",
            f"- Issue bundle reproducers executed: {issue_bundle.get('executed_reproducer_count', 0)}/{issue_bundle.get('extracted_reproducer_count', 0)}",
            f"- Issue bundle reproducer attempts: {issue_bundle.get('executed_reproducer_attempt_count', 0)}",
            f"- Issue bundle flaky reproducers: {issue_bundle.get('flaky_reproducer_count', 0)}",
            f"- Issue bundle clean execution: {str(issue_bundle.get('clean_execution', False)).lower()}",
            "- Issue bundle failures: missing={missing}, compile={compile}, flaky={flaky}, nonzero={nonzero}, timeout={timeout}".format(
                missing=issue_bundle.get("missing_reproducer_count", 0),
                compile=issue_bundle.get("compile_failure_count", 0),
                flaky=issue_bundle.get("flaky_reproducer_count", 0),
                nonzero=issue_bundle.get("nonzero_exit_count", 0),
                timeout=issue_bundle.get("timeout_count", 0),
            ),
            "",
            "## Evidence Chain",
            "",
            f"- Manifest: `{evidence_chain.get('manifest', report['manifest_file'])}`",
            f"- Experiment summary: `{evidence_chain.get('experiment_summary_markdown', summary_md)}`",
            f"- Run CSV: `{evidence_chain.get('run_csv', run_csv)}`",
            f"- Aggregate CSV: `{evidence_chain.get('aggregate_csv', aggregate_csv)}`",
            f"- Aggregate JSON: `{evidence_chain.get('aggregate_json', aggregate_json)}`",
            f"- Run logs indexed: {len(evidence_chain.get('run_logs', []))}",
            f"- Run-provenance freeze manifests indexed: {len(run_provenance.get('freeze_manifest_paths', []))}",
            f"- Run-provenance pip-freeze artifacts indexed: {len(run_provenance.get('freeze_pip_freeze_paths', []))}",
            f"- Run-provenance git-status artifacts indexed: {len(run_provenance.get('freeze_git_status_paths', []))}",
            f"- Run-provenance git-diff artifacts indexed: {len(run_provenance.get('freeze_git_diff_paths', []))}",
            f"- Run-provenance launcher-env artifacts indexed: {len(run_provenance.get('freeze_launcher_env_paths', []))}",
            f"- Run-provenance strategy-snapshot artifacts indexed: {len(run_provenance.get('freeze_strategy_snapshot_paths', []))}",
            f"- Run-provenance strategy-learning artifacts indexed: {len(run_provenance.get('freeze_strategy_learning_paths', []))}",
            f"- Artifact dirs indexed: {len(evidence_chain.get('artifact_dirs', []))}",
            f"- Issue bundle manifest: `{evidence_chain.get('issue_bundle_manifest', '')}`",
            f"- Issue bundle reproducers indexed: {len(evidence_chain.get('issue_bundle_reproducers', []))}",
            f"- Bug-sprint manifests indexed: {len(evidence_chain.get('bug_sprint_manifests', []))}",
            f"- Candidate pipeline manifests indexed: {len(evidence_chain.get('candidate_pipeline_manifests', []))}",
            "",
            "## Ablation And Comparisons",
            "",
            f"- Reference groups: {ablation['reference_run_groups']}",
            f"- Ablation groups: {ablation['ablation_run_groups']} ({', '.join(ablation['ablation_variant_ids']) or 'none'})",
            f"- Ablation variants: {', '.join(ablation['ablation_variant_ids']) or 'none'}",
            f"- Ablated modules covered: {', '.join(ablation['ablation_modules']) or 'none'}",
            f"- Missing planned ablation modules: {', '.join(ablation['missing_ablation_modules']) or 'none'}",
            f"- Ablation candidate bug cases: {ablation['ablation_candidate_bug_cases']}",
            f"- Ablation false positives: {ablation['ablation_false_positive_count']}",
            "",
            "| target suite | comparison group | variant | reference variant | preset | candidate-rate delta | candidate-rate ratio | throughput delta | throughput ratio | evidence bytes/case delta | evidence bytes/case ratio | false-positive delta |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["comparisons"]:
        lines.append(
            "| {target_suite} | {comparison_group} | {variant_label} | {reference_variant} | {preset} | {rate_delta} | {rate_ratio} | {throughput_delta} | {throughput_ratio} | {space_delta} | {space_ratio} | {fp_delta} |".format(
                target_suite=row["target_suite"],
                comparison_group=row.get("comparison_group", ""),
                variant_label=row.get("variant_label", row.get("variant_id", "")),
                reference_variant=row.get("reference_variant_label", ""),
                preset=row["preset"],
                rate_delta=_fmt_signed_percent(row["candidate_bug_case_rate_delta"]),
                rate_ratio=_fmt_optional_ratio(row["candidate_bug_case_rate_ratio"]),
                throughput_delta=_fmt_signed_float(row["throughput_cases_s_delta"]),
                throughput_ratio=_fmt_optional_ratio(row["throughput_cases_s_ratio"]),
                space_delta=_fmt_signed_float(row["evidence_bytes_per_case_delta"]),
                space_ratio=_fmt_optional_ratio(row["evidence_bytes_per_case_ratio"]),
                fp_delta=row["false_positive_delta"],
            )
        )
    return "\n".join(lines) + "\n"


def _global_first_candidate(rows: list[dict[str, str]]) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for row in rows:
        first_s = row.get("first_candidate_bug_elapsed_s")
        first_idx = row.get("first_candidate_bug_case_index")
        if first_s not in (None, ""):
            key = _float(first_s)
        elif first_idx not in (None, ""):
            key = _float(first_idx)
        else:
            continue
        payload = {
            "target_suite": row.get("target_suite", ""),
            "variant_label": row.get("variant_label", ""),
            "preset": row.get("preset", ""),
            "seed": row.get("seed", ""),
            "case_index": first_idx,
            "elapsed_s": first_s,
        }
        if best is None or key < best[0]:
            best = (key, payload)
    return best[1] if best else None


def _run_log_evidence(run_rows: list[dict[str, str]]) -> dict[str, Any]:
    buckets: Counter[str] = Counter()
    total_findings = 0
    artifact_case_count = 0
    artifact_dirs: dict[str, Path] = {}
    first_seen: dict[str, dict[str, Any]] = {}
    scanned = 0
    missing = 0
    for run_row in run_rows:
        run_path = _resolve_existing_path(run_row.get("run_file", ""))
        if run_path is None:
            missing += 1
            continue
        scanned += 1
        known_saturated = _known_saturated_families_for_run(run_path)
        for fallback_idx, item in enumerate(_iter_jsonl(run_path)):
            findings = item.get("findings", []) or []
            total_findings += len(findings)
            buckets.update(offline_finding_buckets(findings, known_saturated))

            bug_dir_text = str(item.get("bug_dir", "") or "").strip()
            if bug_dir_text:
                artifact_case_count += 1
                artifact_dirs.setdefault(bug_dir_text, _resolve_path(bug_dir_text))

            case = item.get("case", {}) if isinstance(item.get("case", {}), dict) else {}
            case_index = item.get("case_index", fallback_idx)
            elapsed_s = item.get("elapsed_s")
            for family in candidate_issue_family_keys(findings, known_saturated):
                payload = {
                    "target_suite": run_row.get("target_suite", ""),
                    "variant_label": run_row.get("variant_label", ""),
                    "preset": run_row.get("preset", ""),
                    "seed": run_row.get("seed", ""),
                    "run_file": str(run_path),
                    "case_index": case_index,
                    "elapsed_s": elapsed_s,
                    "case_id": case.get("case_id", ""),
                    "case_seed": case.get("seed", ""),
                }
                previous = first_seen.get(family)
                if previous is None or _first_seen_key(payload) < _first_seen_key(previous):
                    first_seen[family] = payload
    return {
        "artifact_reproducibility": _artifact_reproducibility_from_dirs(
            artifact_case_count,
            artifact_dirs,
        ),
        "offline_oracle": _offline_oracle_summary_from_buckets(buckets, total_findings),
        "candidate_family_first_seen": dict(
            sorted(first_seen.items(), key=lambda item: _first_seen_key(item[1]))
        ),
        "run_log_scan": {
            "run_logs_total": len(run_rows),
            "run_logs_scanned": scanned,
            "run_logs_missing": missing,
            "run_logs_scan_skipped": False,
        },
    }


def _empty_run_log_evidence(run_rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "artifact_reproducibility": _artifact_reproducibility_from_dirs(0, {}),
        "offline_oracle": _offline_oracle_summary_from_buckets(Counter(), 0),
        "candidate_family_first_seen": {},
        "run_log_scan": {
            "run_logs_total": len(run_rows),
            "run_logs_scanned": 0,
            "run_logs_missing": 0,
            "run_logs_scan_skipped": True,
        },
    }


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_seen_key(payload: dict[str, Any]) -> tuple[float, float, str]:
    elapsed = payload.get("elapsed_s")
    case_index = payload.get("case_index")
    elapsed_key = _float(elapsed) if elapsed not in (None, "") else float("inf")
    case_key = _float(case_index) if case_index not in (None, "") else float("inf")
    return (elapsed_key, case_key, str(payload.get("run_file", "")))


def _offline_oracle_summary_from_buckets(buckets: Counter[str], total_findings: int) -> dict[str, Any]:
    return {
        "classified_findings": total_findings,
        "buckets": {bucket: buckets.get(bucket, 0) for bucket in OFFLINE_ORACLE_BUCKETS},
        "bucket_rates": {
            bucket: count / total_findings if total_findings else 0.0
            for bucket, count in ((bucket, buckets.get(bucket, 0)) for bucket in OFFLINE_ORACLE_BUCKETS)
        },
    }


def _known_saturated_families_for_run(run_path: Path) -> list[str]:
    meta_path = run_meta_path(run_path)
    if not meta_path.is_file():
        return []
    meta = load_json(meta_path)
    config = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
    return list(config.get("known_saturated_bug_families", []) or [])


_run_known_saturated_bug_families = _known_saturated_families_for_run


def _artifact_reproducibility_from_dirs(artifact_case_count: int, artifact_dirs: dict[str, Path]) -> dict[str, Any]:
    existing_dirs = [path for path in artifact_dirs.values() if path.is_dir()]
    total_dirs = len(artifact_dirs)
    core_payload_count = sum(_has_core_artifact_payload(path) for path in existing_dirs)
    reproduce_count = sum(_has_file(path, "reproduce.py") for path in existing_dirs)
    reduced_case_count = sum(_has_file(path, "reduced_case.json") for path in existing_dirs)
    reduced_reproducer_count = sum(_has_file(path, "reproduce_reduced.py") for path in existing_dirs)
    triage_count = sum(_has_file(path, "triage.json") for path in existing_dirs)
    standalone_count = sum(bool(list(path.glob("standalone_*.py"))) for path in existing_dirs)
    missing_artifact_dirs = sorted(
        bug_dir for bug_dir, path in artifact_dirs.items() if not path.exists()
    )
    return {
        "artifact_case_count": artifact_case_count,
        "artifact_dirs": sorted(artifact_dirs),
        "artifact_dirs_total": total_dirs,
        "artifact_dirs_existing": len(existing_dirs),
        "artifact_dirs_missing": len(missing_artifact_dirs),
        "missing_artifact_dirs": missing_artifact_dirs[:20],
        "artifact_dirs_with_core_payload": core_payload_count,
        "artifact_dirs_with_reproduce_script": reproduce_count,
        "artifact_dirs_with_reduced_case": reduced_case_count,
        "artifact_dirs_with_reduced_reproducer": reduced_reproducer_count,
        "artifact_dirs_with_triage_report": triage_count,
        "artifact_dirs_with_standalone_reproducer": standalone_count,
        "artifact_core_payload_coverage": core_payload_count / total_dirs if total_dirs else 0.0,
        "artifact_reproducer_coverage": reproduce_count / total_dirs if total_dirs else 0.0,
    }


def _run_provenance_reproducibility(run_rows: list[dict[str, str]]) -> dict[str, Any]:
    freeze_artifact_nonempty_counts = {name: 0 for name in RUN_PROVENANCE_ARTIFACT_FIELDS}
    freeze_artifact_present_counts = {name: 0 for name in RUN_PROVENANCE_ARTIFACT_FIELDS}
    freeze_artifact_paths = {name: set() for name in RUN_PROVENANCE_ARTIFACT_FIELDS}
    missing_labels: list[str] = []
    incomplete_live_authority_labels: list[str] = []
    git_commits: set[str] = set()
    run_count_with_provenance = 0
    authority_run_count = 0
    freeze_intent_run_count = 0
    latest_code_claim_run_count = 0
    clean_workspace_run_count = 0
    live_run_count = 0
    live_runs_with_provenance = 0
    qualified_live_authority_run_count = 0

    for row in run_rows:
        label = _run_row_label(row)
        is_live = str(row.get("evidence_mode", "") or "").strip() == "live"
        if is_live:
            live_run_count += 1
        run_path = _resolve_existing_path(row.get("run_file", ""))
        if run_path is None:
            missing_labels.append(label)
            if is_live:
                incomplete_live_authority_labels.append(label)
            continue
        meta_path = run_meta_path(run_path)
        if not meta_path.is_file():
            missing_labels.append(label)
            if is_live:
                incomplete_live_authority_labels.append(label)
            continue
        meta = load_json(meta_path)
        provenance = meta.get("run_provenance", {}) if isinstance(meta.get("run_provenance", {}), dict) else {}
        if not provenance:
            missing_labels.append(label)
            if is_live:
                incomplete_live_authority_labels.append(label)
            continue
        run_count_with_provenance += 1
        if is_live:
            live_runs_with_provenance += 1
        vcs = provenance.get("vcs", {}) if isinstance(provenance.get("vcs", {}), dict) else {}
        harness = provenance.get("harness", {}) if isinstance(provenance.get("harness", {}), dict) else {}
        freeze_artifacts = (
            provenance.get("freeze_artifacts", {}) if isinstance(provenance.get("freeze_artifacts", {}), dict) else {}
        )
        git_commit = str(vcs.get("git_commit", "") or "").strip()
        if git_commit:
            git_commits.add(git_commit)
        authority = bool(harness.get("authority", False))
        freeze_intent = bool(harness.get("freeze_intent", False))
        latest_code_claim = bool(harness.get("latest_code_claim", False))
        clean_workspace = vcs.get("workspace_dirty") is False
        authority_run_count += int(authority)
        freeze_intent_run_count += int(freeze_intent)
        latest_code_claim_run_count += int(latest_code_claim)
        clean_workspace_run_count += int(clean_workspace)

        run_artifact_present: dict[str, bool] = {}
        for artifact_name in RUN_PROVENANCE_ARTIFACT_FIELDS:
            artifact_text = str(freeze_artifacts.get(artifact_name, "") or "").strip()
            if artifact_text:
                freeze_artifact_nonempty_counts[artifact_name] += 1
                freeze_artifact_path = _resolve_path(artifact_text)
                if freeze_artifact_path.is_file():
                    freeze_artifact_present_counts[artifact_name] += 1
                    freeze_artifact_paths[artifact_name].add(str(freeze_artifact_path))
                    run_artifact_present[artifact_name] = True
                else:
                    run_artifact_present[artifact_name] = False
            else:
                run_artifact_present[artifact_name] = False

        qualified_live_authority = (
            is_live
            and authority
            and freeze_intent
            and latest_code_claim
            and clean_workspace
            and bool(git_commit)
            and all(run_artifact_present[name] for name in RUN_PROVENANCE_REQUIRED_FREEZE_ARTIFACTS)
        )
        if qualified_live_authority:
            qualified_live_authority_run_count += 1
        elif is_live:
            incomplete_live_authority_labels.append(label)

    return {
        "run_logs_total": len(run_rows),
        "run_count_with_provenance": run_count_with_provenance,
        "run_provenance_coverage": run_count_with_provenance / len(run_rows) if run_rows else 0.0,
        "live_run_count": live_run_count,
        "live_runs_with_provenance": live_runs_with_provenance,
        "authority_run_count": authority_run_count,
        "freeze_intent_run_count": freeze_intent_run_count,
        "latest_code_claim_run_count": latest_code_claim_run_count,
        "clean_workspace_run_count": clean_workspace_run_count,
        "qualified_live_authority_run_count": qualified_live_authority_run_count,
        "all_live_runs_have_qualified_authority_provenance": (
            live_run_count > 0 and qualified_live_authority_run_count == live_run_count
        ),
        "git_commit_count": len(git_commits),
        "git_commits": sorted(git_commits),
        "single_git_commit": len(git_commits) == 1 if git_commits else False,
        "freeze_artifact_nonempty_counts": freeze_artifact_nonempty_counts,
        "freeze_artifact_present_counts": freeze_artifact_present_counts,
        "freeze_manifest_paths": sorted(freeze_artifact_paths["manifest"]),
        "freeze_pip_freeze_paths": sorted(freeze_artifact_paths["pip_freeze"]),
        "freeze_git_status_paths": sorted(freeze_artifact_paths["git_status"]),
        "freeze_git_diff_paths": sorted(freeze_artifact_paths["git_diff"]),
        "freeze_launcher_env_paths": sorted(freeze_artifact_paths["launcher_env"]),
        "freeze_strategy_snapshot_paths": sorted(freeze_artifact_paths["strategy_snapshot"]),
        "freeze_strategy_learning_paths": sorted(freeze_artifact_paths["strategy_learning"]),
        "missing_labels": sorted(set(missing_labels)),
        "incomplete_live_authority_labels": sorted(set(incomplete_live_authority_labels)),
    }


def _issue_bundle_reproducibility(manifest_file: Path) -> dict[str, Any]:
    path = _issue_bundle_manifest_path(manifest_file)
    if path is None:
        default_path = _issue_bundle_manifest_candidates(manifest_file)[0]
        return _empty_issue_bundle_reproducibility(default_path)
    data = load_json(path)
    if not isinstance(data, dict):
        return _empty_issue_bundle_reproducibility(path)
    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    inputs = data.get("inputs", {}) if isinstance(data.get("inputs"), dict) else {}
    issues = data.get("issues", []) if isinstance(data.get("issues"), list) else []
    extracted = _int(summary.get("extracted_reproducer_count"))
    executed = _int(summary.get("executed_reproducer_count"))
    missing = _int(summary.get("missing_reproducer_count"))
    compile_failures = _int(summary.get("compile_failure_count"))
    nonzero = _int(summary.get("nonzero_exit_count"))
    timeouts = _int(summary.get("timeout_count"))
    flaky = _int(summary.get("flaky_reproducer_count"))
    attempts = _int(summary.get("executed_reproducer_attempt_count")) or executed
    repeat_count = _int(inputs.get("repeat_count"))
    run_reproducers = bool(inputs.get("run_reproducers"))
    all_executed = extracted > 0 and executed >= extracted
    clean_execution = bool(
        run_reproducers
        and all_executed
        and missing == 0
        and compile_failures == 0
        and flaky == 0
        and nonzero == 0
        and timeouts == 0
    )
    return {
        "path": str(path),
        "present": True,
        "schema_version": str(data.get("schema_version", "")),
        "generated_at": str(data.get("generated_at", "")),
        "generated_by": str(data.get("generated_by", "")),
        "run_reproducers": run_reproducers,
        "repeat_count": repeat_count,
        "family_count": _int(summary.get("family_count")),
        "families": list(summary.get("families", []) or []),
        "issue_count": _int(summary.get("issue_count")),
        "extracted_reproducer_count": extracted,
        "missing_reproducer_count": missing,
        "compile_failure_count": compile_failures,
        "executed_reproducer_count": executed,
        "executed_reproducer_attempt_count": attempts,
        "flaky_reproducer_count": flaky,
        "nonzero_exit_count": nonzero,
        "nonzero_exit_attempt_count": _int(summary.get("nonzero_exit_attempt_count")),
        "timeout_count": timeouts,
        "timeout_attempt_count": _int(summary.get("timeout_attempt_count")),
        "executed_reproducer_coverage": executed / extracted if extracted else 0.0,
        "all_selected_reproducers_executed": all_executed,
        "clean_execution": clean_execution,
        "issue_paths": [str(issue.get("issue_path", "")) for issue in issues if issue.get("issue_path")],
        "reproducer_paths": [
            str(issue.get("reproducer_path", "")) for issue in issues if issue.get("reproducer_path")
        ],
    }


def _empty_issue_bundle_reproducibility(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": False,
        "schema_version": "",
        "generated_at": "",
        "generated_by": "",
        "run_reproducers": False,
        "repeat_count": 0,
        "family_count": 0,
        "families": [],
        "issue_count": 0,
        "extracted_reproducer_count": 0,
        "missing_reproducer_count": 0,
        "compile_failure_count": 0,
        "executed_reproducer_count": 0,
        "executed_reproducer_attempt_count": 0,
        "flaky_reproducer_count": 0,
        "nonzero_exit_count": 0,
        "nonzero_exit_attempt_count": 0,
        "timeout_count": 0,
        "timeout_attempt_count": 0,
        "executed_reproducer_coverage": 0.0,
        "all_selected_reproducers_executed": False,
        "clean_execution": False,
        "issue_paths": [],
        "reproducer_paths": [],
    }


def _workflow_generated_dir(manifest_file: Path) -> Path:
    candidates = [
        PROJECT_ROOT / "new_issue" / "generated",
        manifest_file.resolve().parent.parent / "new_issue" / "generated",
        Path.cwd() / "new_issue" / "generated",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def _scheduler_effectiveness(generated_dir: Path) -> dict[str, Any]:
    if not generated_dir.is_dir():
        return {
            "manifest_count": 0,
            "manifest_paths": [],
            "completed_run_count": 0,
            "lane_score_sample_count": 0,
            "avg_score": 0.0,
            "avg_budget_multiplier": 0.0,
            "lane_yield": [],
        }
    paths = sorted(generated_dir.glob("bug-sprint*-manifest.json"))
    lane_stats: dict[str, dict[str, Any]] = {}
    total_score = 0.0
    total_budget = 0.0
    score_samples = 0
    completed_runs = 0
    for path in paths:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        for run in data.get("runs", []) or []:
            if not isinstance(run, dict) or str(run.get("status", "")) != "completed":
                continue
            lane_id = str(run.get("lane_id", ""))
            if not lane_id:
                continue
            completed_runs += 1
            stats = lane_stats.setdefault(
                lane_id,
                {
                    "lane_id": lane_id,
                    "completed_runs": 0,
                    "fresh_candidate_total": 0,
                    "unique_fresh_families": set(),
                    "false_positive_total": 0,
                    "score_total": 0.0,
                    "budget_total": 0.0,
                    "yield_total": 0.0,
                    "novelty_total": 0.0,
                    "false_positive_rate_total": 0.0,
                    "score_samples": 0,
                },
            )
            classification = run.get("classification", {}) if isinstance(run.get("classification"), dict) else {}
            fresh = Counter(classification.get("fresh_candidate_bug_families", {}) or {})
            false_positive = Counter(classification.get("false_positive_reasons", {}) or {})
            scheduler = run.get("scheduler", {}) if isinstance(run.get("scheduler"), dict) else {}
            stats["completed_runs"] += 1
            stats["fresh_candidate_total"] += sum(fresh.values())
            stats["unique_fresh_families"].update(fresh)
            stats["false_positive_total"] += sum(false_positive.values())
            if scheduler:
                stats["score_total"] += _float(scheduler.get("score"))
                stats["budget_total"] += _float(scheduler.get("budget_multiplier"))
                stats["yield_total"] += _float(scheduler.get("yield_rate"))
                stats["novelty_total"] += _float(scheduler.get("novelty_rate"))
                stats["false_positive_rate_total"] += _float(scheduler.get("false_positive_rate"))
                stats["score_samples"] += 1
                total_score += _float(scheduler.get("score"))
                total_budget += _float(scheduler.get("budget_multiplier"))
                score_samples += 1
    lane_rows = []
    for lane_id, stats in lane_stats.items():
        samples = max(1, int(stats["score_samples"]))
        lane_rows.append(
            {
                "lane_id": lane_id,
                "completed_runs": int(stats["completed_runs"]),
                "fresh_candidate_total": int(stats["fresh_candidate_total"]),
                "unique_fresh_family_count": len(stats["unique_fresh_families"]),
                "false_positive_total": int(stats["false_positive_total"]),
                "avg_score": stats["score_total"] / samples if stats["score_samples"] else 0.0,
                "avg_budget_multiplier": stats["budget_total"] / samples if stats["score_samples"] else 0.0,
                "avg_yield_rate": stats["yield_total"] / samples if stats["score_samples"] else 0.0,
                "avg_novelty_rate": stats["novelty_total"] / samples if stats["score_samples"] else 0.0,
                "avg_false_positive_rate": (
                    stats["false_positive_rate_total"] / samples if stats["score_samples"] else 0.0
                ),
            }
        )
    lane_rows.sort(
        key=lambda row: (-row["avg_score"], -row["fresh_candidate_total"], row["lane_id"])
    )
    return {
        "manifest_count": len(paths),
        "manifest_paths": [str(path) for path in paths],
        "completed_run_count": completed_runs,
        "lane_score_sample_count": score_samples,
        "avg_score": total_score / score_samples if score_samples else 0.0,
        "avg_budget_multiplier": total_budget / score_samples if score_samples else 0.0,
        "lane_yield": lane_rows,
    }


def _adaptive_scheduler_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    adaptive_state = manifest.get("adaptive_state", []) or []
    arm_rows: list[dict[str, Any]] = []
    reward_signals: list[float] = []
    mean_rewards: list[float] = []
    for item in adaptive_state:
        if not isinstance(item, dict):
            continue
        arm_id = str(item.get("arm_id", "") or "")
        if not arm_id:
            continue
        reward_signal = _float(item.get("reward_signal"))
        mean_reward = _float(item.get("mean_reward"))
        reward_signals.append(reward_signal)
        mean_rewards.append(mean_reward)
        arm_rows.append(
            {
                "arm_id": arm_id,
                "target_suite": str(item.get("target_suite", "") or ""),
                "pulls": int(item.get("pulls", 0) or 0),
                "mean_reward": mean_reward,
                "reward_signal": reward_signal,
                "last_reward": _float(item.get("last_reward")),
                "stale_batches": int(item.get("stale_batches", 0) or 0),
            }
        )
    arm_rows.sort(
        key=lambda row: (
            -float(row.get("reward_signal", 0.0)),
            -float(row.get("mean_reward", 0.0)),
            -int(row.get("pulls", 0)),
            str(row.get("arm_id", "")),
        )
    )
    return {
        "adaptive_enabled": manifest.get("schedule") == "adaptive",
        "local_source_scheduler": manifest.get("local_source_scheduler", {"enabled": False}),
        "adaptive_final_arm_count": len(arm_rows),
        "adaptive_avg_reward_signal": sum(reward_signals) / len(reward_signals) if reward_signals else 0.0,
        "adaptive_max_reward_signal": max(reward_signals) if reward_signals else 0.0,
        "adaptive_avg_mean_reward": sum(mean_rewards) / len(mean_rewards) if mean_rewards else 0.0,
        "adaptive_final_arms": arm_rows[:8],
    }


def _candidate_pipeline_metrics(generated_dir: Path) -> dict[str, Any]:
    pipeline_root = generated_dir / "candidate-pipelines"
    if not pipeline_root.is_dir():
        return {
            "manifest_count": 0,
            "manifest_paths": [],
            "candidate_count": 0,
            "rechecked_count": 0,
            "reproduced_count": 0,
            "recheck_pass_rate": 0.0,
            "reduced_count": 0,
            "candidate_bug_verdict_count": 0,
            "issue_draft_count": 0,
            "needs_dedup_check_count": 0,
            "already_submitted_or_confirmed_count": 0,
            "strategy_snapshot_count": 0,
            "strategy_learning_count": 0,
        }
    paths = sorted(pipeline_root.glob("*/manifest.json"))
    aggregate: Counter[str] = Counter()
    strategy_snapshot_paths: list[str] = []
    strategy_learning_paths: list[str] = []
    for path in paths:
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        aggregate.update(
            {
                "candidate_count": int(summary.get("candidate_count", 0) or 0),
                "rechecked_count": int(summary.get("rechecked_count", 0) or 0),
                "reproduced_count": int(summary.get("reproduced_count", 0) or 0),
                "reduced_count": int(summary.get("reduced_count", 0) or 0),
                "candidate_bug_verdict_count": int(summary.get("candidate_bug_verdict_count", 0) or 0),
                "issue_draft_count": int(summary.get("issue_draft_count", 0) or 0),
                "needs_dedup_check_count": int(summary.get("needs_dedup_check_count", 0) or 0),
                "already_submitted_or_confirmed_count": int(
                    summary.get("already_submitted_or_confirmed_count", 0) or 0
                ),
            }
        )
        snapshot_path = str(data.get("strategy_snapshot_path", "") or "")
        if snapshot_path:
            strategy_snapshot_paths.append(snapshot_path)
        for candidate in data.get("candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            learning_path = str(candidate.get("strategy_learning_path", "") or "")
            if learning_path:
                strategy_learning_paths.append(learning_path)
    return {
        "manifest_count": len(paths),
        "manifest_paths": [str(path) for path in paths],
        **dict(aggregate),
        "recheck_pass_rate": (
            aggregate["reproduced_count"] / aggregate["candidate_count"] if aggregate["candidate_count"] else 0.0
        ),
        "strategy_snapshot_count": len(sorted(set(strategy_snapshot_paths))),
        "strategy_learning_count": len(sorted(set(strategy_learning_paths))),
        "strategy_snapshot_paths": sorted(set(strategy_snapshot_paths)),
        "strategy_learning_paths": sorted(set(strategy_learning_paths)),
    }


def _issue_bundle_manifest_path(manifest_file: Path) -> Path | None:
    for path in _issue_bundle_manifest_candidates(manifest_file):
        if path.is_file():
            return path
    return None


def _issue_bundle_manifest_candidates(manifest_file: Path) -> list[Path]:
    resolved = manifest_file.resolve()
    suffix = Path("new_issue") / "generated" / "issue-bundles" / "manifest.json"
    candidates = [
        resolved.parent.parent / suffix,
        Path.cwd() / suffix,
        resolved.parent / suffix,
    ]
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        key = path.resolve() if path.exists() else path.absolute()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _resolve_existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = _resolve_path(value)
    return path if path.exists() else None


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    return Path.cwd() / path


def _run_row_label(row: dict[str, str]) -> str:
    return ":".join(
        [
            str(row.get("evidence_mode", "") or "").strip() or "unknown",
            str(row.get("target_suite", "") or "").strip() or "unknown",
            str(row.get("preset", "") or "").strip() or "unknown",
            str(row.get("seed", "") or "").strip() or "unknown",
        ]
    )


def _has_core_artifact_payload(path: Path) -> bool:
    return all(
        _has_file(path, name)
        for name in ("case.json", "results.json", "normalized.json", "findings.json", "config.json")
    )


def _has_file(path: Path, name: str) -> bool:
    return (path / name).is_file()


def _parse_counter_summary(value: str) -> Counter:
    out: Counter = Counter()
    if not value or value == "none":
        return out
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        key, count_text = chunk.rsplit(":", 1)
        try:
            out[key.strip()] += int(float(count_text.strip()))
        except ValueError:
            continue
    return out


def _path_size(value: str) -> int:
    if not value:
        return 0
    path = Path(value)
    return path.stat().st_size if path.exists() else 0


def _int_or_path_size(value: Any, path_value: str) -> int:
    parsed = _int(value)
    return parsed if parsed else _path_size(path_value)


def _avg_float(rows: list[dict[str, str]], key: str) -> float:
    values = [_float(row.get(key)) for row in rows if row.get(key) not in (None, "")]
    return sum(values) / len(values) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _fmt_float(value: float) -> str:
    return f"{value:.2f}"


def _fmt_signed_float(value: float) -> str:
    return f"{value:+.2f}"


def _fmt_percent(value: float) -> str:
    return f"{value:.1%}"


def _fmt_signed_percent(value: float) -> str:
    return f"{value:+.1%}"


def _fmt_optional_number(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_optional_ratio(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}x"


def _counter_text(counter: dict[str, Any]) -> str:
    return ", ".join(f"{key}:{value}" for key, value in sorted(counter.items())) or "none"
