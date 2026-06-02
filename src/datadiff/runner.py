from __future__ import annotations

from collections import Counter
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from datadiff.artifact import save_issue_artifact as save_bug_artifact
from datadiff.adjudication import build_adjudication, counts_as_bug_evidence
from datadiff.backends import make_backend
from datadiff.backends.base import Backend
from datadiff.case_features import PROBE_ROOTS
from datadiff.case_policy import known_replay_source_filter_reason
from datadiff.canonicalization import short_canonical_hash
from datadiff.classification_oracle import annotate_findings
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES, ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.dsl import Case
from datadiff.env import collect_environment
from datadiff.feedback import FeedbackState
from datadiff.guidance import GuidanceState
from datadiff.metamorphic import build_metamorphic_variants, evaluate_metamorphic_variants
from datadiff.normalizer import normalize_result
from datadiff.oracle import Finding, evaluate_case
from datadiff.operation_combo import combo_semantic_signals, describe_operation_combo
from datadiff.operation_semantics import op_kind, operation_names
from datadiff.preflight import preflight_case
from datadiff.quality_oracles import evaluate_quality_oracles
from datadiff.reward import (
    analyze_finding_outcomes,
    source_reward_adjustment_from_summary,
    offline_finding_bucket,
    row_reward_signals,
    feedback_summary_for_case,
)
from datadiff.run_provenance import collect_run_provenance
from datadiff.scheduler import LocalSourceScheduler
from datadiff.semantic_signal import canonical_target_key, legacy_target_key_alias, semantic_signal_feature
from datadiff.targets import describe_targets, target_context
from datadiff.util import (
    CORPUS_DIR,
    RUNS_DIR,
    JsonlWriter,
    append_jsonl,
    closed_loop_state_path,
    dump_json,
    ensure_dirs,
    run_meta_path,
    utc_now,
)

ProgressCallback = Callable[[dict[str, Any]], None]
STAGE_PROFILE_KEYS = (
    "generate_mutate_ms",
    "backend_execution_ms",
    "normalize_ms",
    "oracle_classification_ms",
    "scheduler_feedback_ms",
    "logging_artifact_ms",
    "total_case_wall_ms",
)
RUN_LOG_BUFFER_LINES = 32
CASE_LOG_BUFFER_LINES = 16


def _effective_guidance_targets(config: ExperimentConfig) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in _configured_guidance_targets(config):
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _configured_guidance_targets(config: ExperimentConfig) -> list[str]:
    configured: list[str] = []
    seen: set[str] = set()

    def _append(value: Any) -> None:
        text = str(value).strip()
        if not text or text in seen:
            return
        seen.add(text)
        configured.append(text)

    for value in config.guidance_targets or []:
        _append(value)
    for value in config.semantic_focus_families or []:
        family = str(value).strip()
        if not family:
            continue
        _append(f"semantic_family:{family}")
    for value in config.semantic_focus_signals or []:
        signal = str(value).strip()
        if not signal:
            continue
        semantic_target = str(canonical_target_key(semantic_signal_feature(signal))).strip()
        if semantic_target:
            _append(semantic_target)
    return configured


def _config_payload_with_effective_guidance_targets(config: ExperimentConfig) -> dict[str, Any]:
    payload = config.to_dict()
    payload["effective_guidance_targets"] = _configured_guidance_targets(config)
    return payload


def _selected_candidate_metadata(
    case: Case,
    config: ExperimentConfig,
    guidance_enabled: bool,
) -> dict[str, Any]:
    generated_metadata = _generated_candidate_metadata(case)
    return {
        "source": "generated",
        "generated_seed": case.seed,
        "seed_lineage": generated_metadata["seed_lineage"],
        "mutation": generated_metadata["mutation"],
        "feedback_selection": {},
        "feedback_decision": {},
        "operation_combo": describe_operation_combo(case.program.operations),
        "preflight": {
            "valid": True,
            "repaired": False,
            "fallback_used": False,
            "errors_before": [],
            "errors_after": [],
        },
        "replay_filter": {
            "enabled": not config.enable_replay_bug,
            "filtered_before_candidate": 0,
            "fallback_used": False,
            "last_skip_reason": "",
        },
        "family_saturation_filter": {
            "enabled": guidance_enabled,
            "filtered_before_candidate": 0,
            "fallback_used": False,
            "last_skip_reason": "",
        },
    }


def _case_summary(case_data: dict[str, Any]) -> dict[str, Any]:
    program = case_data.get("program", {})
    return {
        "case_id": case_data.get("case_id", ""),
        "seed": case_data.get("seed", 0),
        "table_count": len(case_data.get("tables", [])),
        "row_count": sum(len(table.get("rows", [])) for table in case_data.get("tables", [])),
        "program": {
            "program_id": program.get("program_id", ""),
            "seed": program.get("seed", 0),
            "operations": program.get("operations", []),
        },
    }


def _normalized_summary(normalized: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        backend: {
            "backend": data.get("backend", backend),
            "status": data.get("status", "unknown"),
            "columns": data.get("columns", []),
            "row_count": len(data.get("rows", [])),
            "error_type": data.get("error_type", ""),
            "error": data.get("error", ""),
        }
        for backend, data in normalized.items()
    }


def _raw_results_summary(raw_results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        backend: {
            "backend": data.get("backend", backend),
            "status": data.get("status", "unknown"),
            "error_type": data.get("error_type", ""),
            "duration_ms": data.get("duration_ms", 0.0),
        }
        for backend, data in raw_results.items()
    }


def _empty_stage_profile() -> dict[str, float]:
    return {key: 0.0 for key in STAGE_PROFILE_KEYS}


def _stage_profile_with_total(stage_profile: dict[str, float]) -> dict[str, float]:
    profile = _empty_stage_profile()
    for key in STAGE_PROFILE_KEYS:
        if key == "total_case_wall_ms":
            continue
        profile[key] = float(stage_profile.get(key, 0.0) or 0.0)
    profile["total_case_wall_ms"] = sum(
        profile[key]
        for key in STAGE_PROFILE_KEYS
        if key != "total_case_wall_ms"
    )
    return profile


def _merge_stage_profile(totals: dict[str, float], stage_profile: dict[str, float]) -> dict[str, float]:
    merged = _empty_stage_profile()
    for key in STAGE_PROFILE_KEYS:
        merged[key] = float(totals.get(key, 0.0) or 0.0) + float(stage_profile.get(key, 0.0) or 0.0)
    return merged


def _finalize_stage_profile_summary(totals: dict[str, float], *, cases: int) -> dict[str, Any]:
    summary = {
        "totals_ms": _stage_profile_with_total(totals),
        "avg_ms_per_case": _empty_stage_profile(),
        "share_of_total": _empty_stage_profile(),
        "case_count": max(0, int(cases)),
    }
    case_count = max(0, int(cases))
    total_wall_ms = float(summary["totals_ms"].get("total_case_wall_ms", 0.0) or 0.0)
    for key in STAGE_PROFILE_KEYS:
        total_ms = float(summary["totals_ms"].get(key, 0.0) or 0.0)
        summary["avg_ms_per_case"][key] = total_ms / case_count if case_count else 0.0
        summary["share_of_total"][key] = total_ms / total_wall_ms if total_wall_ms else 0.0
    return summary


def _guidance_summary(guidance: dict[str, Any]) -> dict[str, Any]:
    family_saturation_penalty = guidance.get("score_breakdown", {}).get("family_saturation_penalty", 0.0)
    family_saturation_active = guidance.get("score_breakdown", {}).get("family_saturation_active", 0.0)
    return {
        "strategy": guidance.get("strategy", ""),
        "score": guidance.get("score", 0.0),
        "matched_targets": guidance.get("matched_targets", []),
        "matched_semantic_targets": guidance.get("matched_targets", []),
        "candidate_count": guidance.get("candidate_count", 1),
        "contributing_candidate_count": guidance.get("contributing_candidate_count", guidance.get("candidate_count", 1)),
        "pruned_candidate_count": guidance.get("pruned_candidate_count", 0),
        "feature_count": len(guidance.get("features", [])),
        "frontier_bucket_count": len(guidance.get("frontier_buckets", [])),
        "discovery_bucket_count": len(guidance.get("discovery_buckets", [])),
        "path_coverage_proxy": guidance.get("score_breakdown", {}).get("path_coverage_proxy", 0.0),
        "data_sensitivity": guidance.get("score_breakdown", {}).get("data_sensitivity", 0.0),
        "frontier_conformance": guidance.get("score_breakdown", {}).get("frontier_conformance", 0.0),
        "discovery_diversity_bonus": guidance.get("score_breakdown", {}).get("discovery_diversity_bonus", 0.0),
        "candidate_pool_diversity_bonus": guidance.get("score_breakdown", {}).get(
            "candidate_pool_diversity_bonus", 0.0
        ),
        "discovery_stale_penalty": guidance.get("score_breakdown", {}).get("discovery_stale_penalty", 0.0),
        "discovery_stale_active": guidance.get("score_breakdown", {}).get("discovery_stale_active", 0.0),
        "recent_discovery_loop_penalty": guidance.get("score_breakdown", {}).get(
            "recent_discovery_loop_penalty", 0.0
        ),
        "recent_discovery_loop_active": guidance.get("score_breakdown", {}).get(
            "recent_discovery_loop_active", 0.0
        ),
        "recent_discovery_window_count": guidance.get("score_breakdown", {}).get(
            "recent_discovery_window_count", 0.0
        ),
        "contribution_potential": guidance.get("score_breakdown", {}).get("contribution_potential", 0.0),
        "combo_priority": guidance.get("score_breakdown", {}).get("combo_priority", 0.0),
        "online_weight_mean": guidance.get("score_breakdown", {}).get("online_weight_mean", 1.0),
        "online_weight_max": guidance.get("score_breakdown", {}).get("online_weight_max", 1.0),
        "online_weight_updates": guidance.get("score_breakdown", {}).get("online_weight_updates", 0.0),
        "resolved_semantic_boundary_penalty": guidance.get("score_breakdown", {}).get(
            "resolved_semantic_boundary_penalty", 0.0
        ),
        "profile_saturation_penalty": guidance.get("score_breakdown", {}).get("profile_saturation_penalty", 0.0),
        "profile_saturation_active": guidance.get("score_breakdown", {}).get("profile_saturation_active", 0.0),
        "family_saturation_penalty": family_saturation_penalty,
        "family_saturation_active": family_saturation_active,
        "family_diversity_guard_penalty": family_saturation_penalty,
        "family_diversity_guard_active": family_saturation_active,
        "issue_replay_saturation_penalty": guidance.get("score_breakdown", {}).get(
            "issue_replay_saturation_penalty", 0.0
        ),
        "issue_replay_saturation_active": guidance.get("score_breakdown", {}).get(
            "issue_replay_saturation_active", 0.0
        ),
        "issue_replay_global_saturation_penalty": guidance.get("score_breakdown", {}).get(
            "issue_replay_global_saturation_penalty", 0.0
        ),
        "issue_replay_global_saturation_active": guidance.get("score_breakdown", {}).get(
            "issue_replay_global_saturation_active", 0.0
        ),
        "issue_inspired_source_saturation_penalty": guidance.get("score_breakdown", {}).get(
            "issue_inspired_source_saturation_penalty", 0.0
        ),
        "issue_inspired_source_saturation_active": guidance.get("score_breakdown", {}).get(
            "issue_inspired_source_saturation_active", 0.0
        ),
    }


def _closed_loop_state_summary(state: dict[str, Any]) -> dict[str, Any]:
    feedback_state = state.get("feedback") if isinstance(state.get("feedback"), dict) else {}
    guidance_state = state.get("guidance") if isinstance(state.get("guidance"), dict) else {}
    return {
        "seen_signature_count": len(state.get("seen_signatures", []) or []),
        "signal_seen_signature_count": len(state.get("signal_seen_signatures", []) or []),
        "feedback_interesting_case_count": len(feedback_state.get("interesting_cases", []) or []),
        "feedback_stored_candidate_family_count": len(feedback_state.get("stored_candidate_bug_families", {}) or {}),
        "feedback_stored_target_key_count": len(feedback_state.get("stored_target_keys", {}) or {}),
        "guidance_feature_count": len(guidance_state.get("feature_counts", {}) or {}),
        "guidance_frontier_bucket_count": len(guidance_state.get("frontier_bucket_counts", {}) or {}),
        "guidance_candidate_bug_family_count": len(guidance_state.get("candidate_bug_family_counts", {}) or {}),
    }


def _quality_oracle_summary(oracles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": oracle.get("name", "unknown"),
            "verdict": oracle.get("verdict", "unknown"),
            "passed": bool(oracle.get("passed", False)),
            "score": oracle.get("score", 0.0),
        }
        for oracle in oracles
    ]


def _generated_candidate_metadata(case: Case) -> dict[str, Any]:
    return {
        "seed_lineage": {
            "root_seed": case.seed,
            "parent_seed": None,
            "parent_case_id": "",
            "mutation_seed": None,
            "depth": 0,
        },
        "mutation": {
            "operator": "generated",
            "detail": "generated",
            "changed": False,
        },
        "feedback_selection": {},
        "feedback_decision": {},
    }


def _source_scheduler_snapshot(feedback: FeedbackState | None) -> list[dict[str, Any]]:
    scheduler = getattr(feedback, "source_scheduler", None) if feedback is not None else None
    if scheduler is None or not hasattr(scheduler, "snapshot"):
        return []
    return scheduler.snapshot()


def _compact_log_row(row: dict[str, Any], log_level: str) -> dict[str, Any]:
    if log_level == "full":
        return row
    has_findings = bool(row.get("findings"))
    if has_findings and log_level in {"compact", "minimal"}:
        # Finding rows keep reproduction detail; run-level duplicated metadata
        # stays in meta.json and artifacts.
        return {
            key: value
            for key, value in row.items()
            if key not in {"environment", "targets"}
        }

    out = {
        "run_at": row.get("run_at", ""),
        "status": row.get("status", "unknown"),
        "case": _case_summary(row.get("case", {})),
        "behavior_signature": row.get("behavior_signature", ""),
        "discovery_signature": row.get("discovery_signature", ""),
        "signal_signature": row.get("signal_signature", ""),
        "duration_ms": row.get("duration_ms", 0.0),
        "stage_profile": _stage_profile_with_total(row.get("stage_profile", {})),
        "findings": row.get("findings", []),
        "bug_dir": row.get("bug_dir", ""),
        "candidate_source": row.get("candidate_source", "generated"),
        "seed_lineage": row.get("seed_lineage", {}),
        "mutation": row.get("mutation", {}),
        "feedback_selection": row.get("feedback_selection", row.get("feedback_decision", {})),
        "feedback_decision": row.get("feedback_decision", {}),
        "operation_combo": row.get("operation_combo", {}),
        "source_reward": row.get("source_reward"),
        "source_scheduler": row.get("source_scheduler", []),
        "preflight": row.get("preflight", {}),
        "quality_oracles": _quality_oracle_summary(row.get("quality_oracles", [])),
        "guidance": _guidance_summary(row.get("guidance", {})),
        "candidate_seed_start": row.get("candidate_seed_start", 0),
        "candidate_pool_size": row.get("candidate_pool_size", 1),
        "case_index": row.get("case_index", 0),
        "elapsed_s": row.get("elapsed_s", 0.0),
        "is_new_behavior": row.get("is_new_behavior", False),
        "signal_new_behavior": row.get("signal_new_behavior", row.get("is_new_behavior", False)),
        "stored_in_feedback_corpus": row.get("stored_in_feedback_corpus", False),
        "feedback_corpus_persisted": row.get("feedback_corpus_persisted", False),
        "feedback_eligible": row.get("feedback_eligible", True),
        "feedback_skip_reason": row.get("feedback_skip_reason", ""),
        "feedback_record_skip_reason": row.get("feedback_record_skip_reason", ""),
        "replay_filter": row.get("replay_filter", {}),
        "family_saturation_filter": row.get("family_saturation_filter", {}),
        "feedback_summary": row.get("feedback_summary", {}),
    }
    if log_level == "compact":
        out["normalized"] = _normalized_summary(row.get("normalized", {}))
        out["raw_results"] = _raw_results_summary(row.get("raw_results", {}))
    else:
        out["backend_status"] = {
            backend: data.get("status", "unknown")
            for backend, data in row.get("normalized", {}).items()
        }
    return out


def behavior_signature(row: dict[str, Any]) -> str:
    payload = {
        "case_ops": row["case"]["program"]["operations"],
        "backend_status": {
            b: r["status"] for b, r in sorted(row["normalized"].items())
        },
        "normalized_shape": {
            b: {
                "columns": r.get("columns", []),
                "rows": len(r.get("rows", [])),
                "sample": r.get("rows", [])[:3],
                "error_type": r.get("error_type", ""),
            }
            for b, r in sorted(row["normalized"].items())
        },
        "finding_kinds": sorted(f["kind"] for f in row.get("findings", [])),
        "finding_roots": sorted(f.get("root_cause", "unknown") for f in row.get("findings", [])),
    }
    return short_canonical_hash(payload, 16)


def _row_count_bucket(row_count: int) -> str:
    if row_count <= 0:
        return "0"
    if row_count == 1:
        return "1"
    if row_count <= 3:
        return "2-3"
    if row_count <= 7:
        return "4-7"
    if row_count <= 15:
        return "8-15"
    if row_count <= 31:
        return "16-31"
    return "32+"


def discovery_signature(row: dict[str, Any]) -> str:
    operations = operation_names(row["case"]["program"]["operations"], default="unknown")
    op_histogram = sorted(Counter(operations).items())
    combo = describe_operation_combo(row["case"]["program"]["operations"])
    findings = row.get("findings") or []
    payload = {
        "combo_template": combo.get("template", ""),
        "operation_histogram": op_histogram,
        "backend_outcomes": {
            backend: {
                "status": result.get("status", "unknown"),
                "error_type": result.get("error_type", ""),
                "column_count": len(result.get("columns", [])),
                "row_count_bucket": _row_count_bucket(len(result.get("rows", []))),
            }
            for backend, result in sorted((row.get("normalized") or {}).items())
        },
        "finding_kinds": sorted(str(finding.get("kind", "")) for finding in findings),
        "finding_roots": sorted(
            str(finding.get("root_cause", "unknown"))
            for finding in findings
            if not bool(finding.get("false_positive"))
        ),
    }
    return short_canonical_hash(payload, 16)


def signal_signature(row: dict[str, Any]) -> str:
    operations = sorted(set(operation_names(row["case"]["program"]["operations"], default="unknown")))
    combo = describe_operation_combo(row["case"]["program"]["operations"])
    findings = row.get("findings") or []
    payload = {
        "combo_template": combo.get("template", ""),
        "operation_set": operations,
        "backend_outcomes": {
            backend: {
                "status": result.get("status", "unknown"),
                "error_type": result.get("error_type", ""),
            }
            for backend, result in sorted((row.get("normalized") or {}).items())
        },
        "finding_buckets": sorted(
            {
                offline_finding_bucket(finding)
                for finding in findings
                if not bool(finding.get("false_positive"))
            }
        ),
        "suspicious_backends": sorted(
            {
                str(backend)
                for finding in findings
                if not bool(finding.get("false_positive"))
                for backend in finding.get("suspicious_backends", []) or []
            }
        ),
    }
    return short_canonical_hash(payload, 16)


CALIBRATION_PROBE_OPS = frozenset(PROBE_ROOTS) | {
    "running_sum",
    "tuple_absence_filter",
}
REPLAY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE = 20
SATURATED_FAMILY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE = 20


def _feedback_storage_decision(
    case: Case,
    *,
    candidate_source: str = "generated",
    seed_lineage: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    lineage_depth = 0
    if isinstance(seed_lineage, dict):
        lineage_depth = int(seed_lineage.get("depth", 0) or 0)
    if candidate_source == "feedback_mutation" or lineage_depth > 0:
        return False, "feedback_mutation_child"
    if any(op_kind(op) in CALIBRATION_PROBE_OPS for op in case.program.operations):
        return False, "calibration_probe_case"
    return True, ""


def _feedback_target_keys(guidance_row: dict[str, Any], operation_combo: dict[str, Any]) -> list[str]:
    broad_targets = {
        "strings",
        "numeric",
        "nulls",
        "mutate",
        "aggregation",
        "expressions",
        "groupby",
        "filter",
        "sort_limit",
        "topk",
        "join",
    }
    keys: list[str] = []
    for target in guidance_row.get("matched_targets", []) or []:
        value = str(target).strip()
        if value and value not in broad_targets:
            keys.append(f"target:{value}")
    for feature in guidance_row.get("features", []) or []:
        value = str(feature).strip()
        if value and (
            value.startswith("pattern:")
            or value.startswith("semantic_family:")
            or value.startswith("source:")
            or value.startswith("profile:")
        ):
            keys.append(f"feature:{value}")
    template = str(operation_combo.get("template", "")).strip()
    if template:
        keys.append(f"combo:{template}")
    for family in guidance_row.get("features", []) or []:
        value = str(family).strip()
        if value.startswith("semantic_family:"):
            keys.append(value)
    for signal in combo_semantic_signals(operation_combo):
        value = str(signal).strip()
        if value:
            semantic_key = semantic_signal_feature(value)
            keys.append(semantic_key)
            legacy_key = legacy_target_key_alias(semantic_key)
            if legacy_key:
                keys.append(legacy_key)
    config_payload = guidance_row.get("config", {}) if isinstance(guidance_row.get("config", {}), dict) else {}
    for family in config_payload.get("semantic_focus_families", []) or []:
        value = str(family).strip()
        if value:
            keys.append(f"semantic_family:{value}")
    for signal in config_payload.get("semantic_focus_signals", []) or []:
        value = str(signal).strip()
        if value:
            semantic_key = semantic_signal_feature(value)
            keys.append(semantic_key)
            legacy_key = legacy_target_key_alias(semantic_key)
            if legacy_key:
                keys.append(legacy_key)
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def _known_replay_source_filter_reason(case_item: Case, config: ExperimentConfig) -> str:
    return known_replay_source_filter_reason(
        case_item,
        enable_replay_bug=config.enable_replay_bug,
        replay_bug_source_issues=config.replay_bug_source_issues,
    )


_replay_bug_filter_reason = _known_replay_source_filter_reason


def _effective_generator_profile(config: ExperimentConfig) -> str:
    return config.generator_profile


def _restore_closed_loop_state(
    closed_loop_state: dict[str, Any] | None,
    *,
    config: ExperimentConfig,
    backends: list[str],
    guidance_enabled: bool,
    feedback_enabled: bool,
) -> tuple[set[str], set[str], FeedbackState | None, GuidanceState | None]:
    state = closed_loop_state if isinstance(closed_loop_state, dict) else {}
    seen = {str(item) for item in state.get("seen_signatures", []) or []}
    signal_seen = {str(item) for item in state.get("signal_seen_signatures", []) or []}
    if not signal_seen:
        signal_seen = set(seen)
    feedback: FeedbackState | None = None
    guidance: GuidanceState | None = None
    if feedback_enabled:
        scheduler_state = None
        if config.enable_local_source_scheduler:
            raw_scheduler_state = state.get("source_scheduler")
            if not isinstance(raw_scheduler_state, dict):
                raw_feedback_state = state.get("feedback")
                if isinstance(raw_feedback_state, dict):
                    raw_scheduler_state = raw_feedback_state.get("source_scheduler")
            if isinstance(raw_scheduler_state, dict):
                scheduler_state = LocalSourceScheduler.from_state_dict(
                    raw_scheduler_state,
                    exploration_weight=config.local_source_exploration_weight,
                    enable_family_saturation=config.enable_family_saturation,
                    family_saturation_threshold=config.family_saturation_threshold,
                    saturated_family_reward=config.saturated_family_reward,
                    known_saturated_bug_families=config.known_saturated_bug_families,
                )
            else:
                scheduler_state = LocalSourceScheduler(
                    exploration_weight=config.local_source_exploration_weight,
                    enable_family_saturation=config.enable_family_saturation,
                    family_saturation_threshold=config.family_saturation_threshold,
                    saturated_family_reward=config.saturated_family_reward,
                    known_saturated_bug_families=config.known_saturated_bug_families,
                )
        raw_feedback_state = state.get("feedback")
        if isinstance(raw_feedback_state, dict):
            feedback = FeedbackState.from_state_dict(
                raw_feedback_state,
                persist_to_disk=config.persist_feedback_corpus,
                max_persisted=config.feedback_persist_limit,
                max_cases_per_profile=config.feedback_max_cases_per_profile,
                source_scheduler=scheduler_state,
            )
        else:
            feedback = FeedbackState(
                persist_to_disk=config.persist_feedback_corpus,
                max_persisted=config.feedback_persist_limit,
                max_cases_per_profile=config.feedback_max_cases_per_profile,
                source_scheduler=scheduler_state,
            )
    if guidance_enabled:
        effective_guidance_targets = _effective_guidance_targets(config)
        raw_guidance_state = state.get("guidance")
        if isinstance(raw_guidance_state, dict):
            guidance = GuidanceState.from_state_dict(
                raw_guidance_state,
                targets=effective_guidance_targets,
                discovery_biases=list(config.discovery_biases),
                enable_family_saturation=config.enable_family_saturation,
                family_saturation_threshold=config.family_saturation_threshold,
                family_saturation_penalty=config.family_saturation_penalty,
                saturated_family_reward=config.saturated_family_reward,
                known_saturated_bug_families=config.known_saturated_bug_families,
                issue_replay_saturation_threshold=config.issue_replay_saturation_threshold,
                issue_replay_saturation_penalty=config.issue_replay_saturation_penalty,
                issue_replay_global_saturation_threshold=config.issue_replay_global_saturation_threshold,
                issue_replay_global_saturation_penalty=config.issue_replay_global_saturation_penalty,
                issue_inspired_source_saturation_threshold=config.issue_inspired_source_saturation_threshold,
                issue_inspired_source_saturation_penalty=config.issue_inspired_source_saturation_penalty,
                active_backends=list(backends),
            )
        else:
            guidance = GuidanceState(
                targets=effective_guidance_targets,
                discovery_biases=list(config.discovery_biases),
                enable_family_saturation=config.enable_family_saturation,
                family_saturation_threshold=config.family_saturation_threshold,
                family_saturation_penalty=config.family_saturation_penalty,
                saturated_family_reward=config.saturated_family_reward,
                known_saturated_bug_families=config.known_saturated_bug_families,
                issue_replay_saturation_threshold=config.issue_replay_saturation_threshold,
                issue_replay_saturation_penalty=config.issue_replay_saturation_penalty,
                issue_replay_global_saturation_threshold=config.issue_replay_global_saturation_threshold,
                issue_replay_global_saturation_penalty=config.issue_replay_global_saturation_penalty,
                issue_inspired_source_saturation_threshold=config.issue_inspired_source_saturation_threshold,
                issue_inspired_source_saturation_penalty=config.issue_inspired_source_saturation_penalty,
                active_backends=list(backends),
            )
    return seen, signal_seen, feedback, guidance


def _build_closed_loop_state(
    *,
    seen: set[str],
    signal_seen: set[str],
    feedback: FeedbackState | None,
    guidance: GuidanceState | None,
) -> dict[str, Any]:
    return {
        "seen_signatures": sorted(seen),
        "signal_seen_signatures": sorted(signal_seen),
        "feedback": feedback.to_state_dict() if feedback is not None else None,
        "source_scheduler": (
            feedback.source_scheduler.to_state_dict()
            if feedback is not None and feedback.source_scheduler is not None
            else None
        ),
        "guidance": guidance.to_state_dict() if guidance is not None else None,
    }


def _execute_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    backend_instances: dict[str, Backend] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    raw_results = {}
    normalized = {}
    for backend_name in backends:
        backend = backend_instances[backend_name] if backend_instances is not None else make_backend(backend_name)
        result = backend.run(case.tables, case.program)
        raw_results[backend_name] = {
            k: v
            for k, v in result.to_dict().items()
            if k != "data"
        }
        normalized[backend_name] = normalize_result(
            result,
            case.program,
            enable_normalizer=config.enable_normalizer,
        )
    return raw_results, normalized


def run_loaded_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig | None = None,
    save_artifact: bool = True,
    backend_instances: dict[str, Backend] | None = None,
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
    config_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    resolved_config_payload = dict(config_payload) if isinstance(config_payload, dict) else _config_payload_with_effective_guidance_targets(config)
    started = time.perf_counter()
    stage_profile = _empty_stage_profile()
    execute_started = time.perf_counter()
    raw_results, normalized = _execute_case(case, backends, config, backend_instances=backend_instances)
    execute_elapsed = (time.perf_counter() - execute_started) * 1000
    stage_profile["backend_execution_ms"] += sum(
        float(result.get("duration_ms", 0.0) or 0.0)
        for result in raw_results.values()
    )
    stage_profile["normalize_ms"] += max(0.0, execute_elapsed - stage_profile["backend_execution_ms"])

    findings = []
    classification_started = time.perf_counter()
    if config.enable_differential_oracle:
        findings = evaluate_case(case, normalized)
    metamorphic_rows: dict[str, Any] = {}
    if config.enable_metamorphic_oracle:
        variant_results = {}
        for variant in build_metamorphic_variants(case, limit=max(0, config.metamorphic_variant_limit)):
            variant_raw, variant_norm = _execute_case(
                variant.case,
                backends,
                config,
                backend_instances=backend_instances,
            )
            variant_results[variant.name] = variant_norm
            metamorphic_rows[variant.name] = {
                "relation": variant.relation,
                "case": variant.case.to_dict(),
                "raw_results": variant_raw,
                "normalized": {k: v.to_dict() for k, v in variant_norm.items()},
            }
        findings.extend(evaluate_metamorphic_variants(case, normalized, variant_results))
    if findings:
        annotate_findings(
            case,
            findings,
            normalized=normalized,
            raw_results=raw_results,
            config=resolved_config_payload,
            backends=backends,
        )
        countable_findings = _countable_finding_objects(findings)
        if countable_findings:
            recheck = _candidate_recheck(case, backends, config, countable_findings)
            countable_findings = _countable_finding_objects(findings)
        else:
            recheck = {
                "enabled": False,
                "attempts": 0,
                "reproduced_keys": [],
                "non_reproduced_keys": [],
                "skip_reason": "no_countable_candidate_findings",
            }
    else:
        countable_findings = []
        recheck = {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}
    stage_profile["oracle_classification_ms"] = (time.perf_counter() - classification_started) * 1000
    stage_profile["total_case_wall_ms"] = (time.perf_counter() - started) * 1000

    row = {
        "run_at": utc_now(),
        "case": case.to_dict(),
        "targets": target_specs if target_specs is not None else describe_targets(backends),
        "raw_results": raw_results,
        "normalized": {k: v.to_dict() for k, v in normalized.items()},
        "metamorphic": metamorphic_rows,
        "findings": [f.to_dict() for f in findings],
        "candidate_recheck": recheck,
        "config": resolved_config_payload,
        "environment": environment if environment is not None else collect_environment(),
        "status": "bug" if countable_findings else "ok",
        "duration_ms": stage_profile["total_case_wall_ms"],
        "stage_profile": stage_profile,
    }
    row["behavior_signature"] = behavior_signature(row)
    row["discovery_signature"] = discovery_signature(row)
    if countable_findings and save_artifact and config.enable_artifact:
        artifact_started = time.perf_counter()
        bug_dir = save_bug_artifact(
            case,
            raw_results=raw_results,
            normalized={k: v.to_dict() for k, v in normalized.items()},
            findings=countable_findings,
            config=resolved_config_payload,
        )
        stage_profile["logging_artifact_ms"] += (time.perf_counter() - artifact_started) * 1000
        stage_profile["total_case_wall_ms"] = sum(
            stage_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row["duration_ms"] = stage_profile["total_case_wall_ms"]
        row["bug_dir"] = str(bug_dir)
    return row


def _candidate_recheck(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    findings: list[Finding],
) -> dict[str, Any]:
    attempts = max(0, int(getattr(config, "candidate_recheck_count", 0)))
    if attempts <= 0 or not findings:
        return {"enabled": False, "attempts": 0, "reproduced_keys": [], "non_reproduced_keys": []}

    recheck_config_data = config.to_dict()
    recheck_config_data["candidate_recheck_count"] = 0
    recheck_config_data["enable_artifact"] = False
    recheck_config = ExperimentConfig(**recheck_config_data)
    original_keys = {_finding_recheck_key(finding.to_dict()) for finding in findings}
    reproduced_keys: set[tuple[str, str, tuple[str, ...], str]] | None = None
    attempt_summaries: list[dict[str, Any]] = []
    for attempt in range(attempts):
        row = run_loaded_case(
            case,
            backends=backends,
            config=recheck_config,
            save_artifact=False,
            backend_instances=None,
            target_specs=[],
        )
        keys = {_finding_recheck_key(finding) for finding in row.get("findings", [])}
        current_reproduced = original_keys & keys
        reproduced_keys = current_reproduced if reproduced_keys is None else reproduced_keys & current_reproduced
        attempt_summaries.append(
            {
                "attempt": attempt + 1,
                "finding_count": len(row.get("findings", []) or []),
                "reproduced_keys": [_format_recheck_key(key) for key in sorted(current_reproduced)],
            }
        )
    reproduced_keys = reproduced_keys or set()
    non_reproduced_keys = original_keys - reproduced_keys
    for finding in findings:
        key = _finding_recheck_key(finding.to_dict())
        if key in non_reproduced_keys:
            _mark_finding_non_reproducible(finding, attempts)
    return {
        "enabled": True,
        "attempts": attempts,
        "attempt_summaries": attempt_summaries,
        "reproduced_keys": [_format_recheck_key(key) for key in sorted(reproduced_keys)],
        "non_reproduced_keys": [_format_recheck_key(key) for key in sorted(non_reproduced_keys)],
    }


def _finding_recheck_key(finding: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
    return (
        str(finding.get("kind", "")),
        str(finding.get("root_cause", "unknown")),
        tuple(sorted(str(backend) for backend in finding.get("suspicious_backends", []) or [])),
        str(finding.get("mismatch_class", "")),
    )


def _format_recheck_key(key: tuple[str, str, tuple[str, ...], str]) -> str:
    kind, root, backends, mismatch = key
    backend_part = ",".join(backends) or "unknown"
    suffix = f":{mismatch}" if mismatch else ""
    return f"{kind}:{root}@{backend_part}{suffix}"


def _mark_finding_non_reproducible(finding: Finding, attempts: int) -> None:
    finding.triage_verdict = "non_reproducible_candidate"
    finding.paper_status = "exclude_unreproducible_candidate"
    finding.triage_confidence = "high"
    finding.false_positive = True
    finding.false_positive_reason = "candidate_not_reproduced_on_immediate_recheck"
    finding.triage_evidence = (
        f"Initial finding did not reproduce in {attempts} immediate fresh recheck run(s); "
        "exclude it from latest-version bug evidence until a stable reproducer exists."
    )
    finding.adjudication = build_adjudication(
        "non_reproducible_candidate",
        validity_gate="recheck_failed",
        semantic_gate="unknown",
        attribution_gate="reproduction_failed",
        recheck_status="failed",
        exclusion_reason="candidate_not_reproduced_on_immediate_recheck",
        needs_manual_review=False,
    )


def run_fuzz(
    cases: int | None,
    seed: int,
    backends: list[str],
    config: ExperimentConfig | None = None,
    duration_s: float | None = None,
    save_cases: bool = False,
    case_log_file: Path | None = None,
    checkpoint_interval_s: float | None = None,
    progress_interval_s: float | None = None,
    progress_callback: ProgressCallback | None = None,
    closed_loop_state: dict[str, Any] | None = None,
    persist_closed_loop_state: bool = False,
) -> Path:
    ensure_dirs()
    config = config or ExperimentConfig()
    if cases is None and duration_s is None:
        cases = 100
    run_id = f"run-{utc_now().replace(':', '').replace('-', '').replace('Z', '')}-{time.time_ns()}"
    run_suffix = ".jsonl.gz" if config.compress_run_log else ".jsonl"
    run_file = RUNS_DIR / f"{run_id}{run_suffix}"
    if case_log_file is not None:
        save_cases = True
    resolved_case_log_file = case_log_file
    if save_cases and resolved_case_log_file is None:
        resolved_case_log_file = CORPUS_DIR / "generated" / f"{run_id}.cases.jsonl"
    checkpoint_file = RUNS_DIR / f"{run_id}.checkpoint.json" if checkpoint_interval_s is not None else None
    backend_instances = {backend_name: make_backend(backend_name) for backend_name in backends}
    environment = collect_environment()
    run_provenance = collect_run_provenance()
    targets = target_context(backends)
    target_specs = targets.target_dicts()
    config_payload = _config_payload_with_effective_guidance_targets(config)
    guidance_targets = _configured_guidance_targets(config)
    guided = config.guidance_strategy == "guided"
    candidate_pool = max(1, config.guidance_candidate_pool if guided else 1)
    seen, signal_seen, feedback, guidance = _restore_closed_loop_state(
        closed_loop_state,
        config=config,
        backends=backends,
        guidance_enabled=guided,
        feedback_enabled=config.enable_feedback,
    )
    started = time.perf_counter()
    last_checkpoint = started
    last_progress = started
    executed = 0
    next_seed = seed
    findings_count = 0
    new_behavior_count = 0
    signal_new_behavior_count = 0
    artifact_saved_count = 0
    stage_profile_totals = _empty_stage_profile()
    preflight_repaired_count = 0
    preflight_fallback_count = 0
    preflight_invalid_count = 0
    replay_filtered_candidate_count = 0
    replay_filter_fallback_count = 0
    saturated_family_filtered_candidate_count = 0
    saturated_family_filter_fallback_count = 0
    quality_oracle_counts: dict[str, int] = {}
    effective_generator_profile = _effective_generator_profile(config)
    persisted_closed_loop_state_path = closed_loop_state_path(run_file) if persist_closed_loop_state else None

    def snapshot(status: str) -> dict[str, Any]:
        elapsed_s = time.perf_counter() - started
        out = {
            "run_id": run_id,
            "status": status,
            "run_file": str(run_file),
            "case_log_file": str(resolved_case_log_file) if resolved_case_log_file is not None else "",
            "checkpoint_file": str(checkpoint_file) if checkpoint_file is not None else "",
            "requested_cases": cases,
            "executed_cases": executed,
            "duration_s": duration_s,
            "elapsed_s": elapsed_s,
            "throughput_cases_s": executed / elapsed_s if elapsed_s else 0.0,
            "findings": findings_count,
            "new_behavior_cases": new_behavior_count,
            "signal_new_behavior_cases": signal_new_behavior_count,
            "saved_artifacts": artifact_saved_count,
            "preflight": {
                "repaired_cases": preflight_repaired_count,
                "fallback_cases": preflight_fallback_count,
                "invalid_cases": preflight_invalid_count,
            },
            "replay_bug_filter": {
                "enabled": not config.enable_replay_bug,
                "filtered_candidates": replay_filtered_candidate_count,
                "fallback_candidates": replay_filter_fallback_count,
            },
            "family_saturation_filter": {
                "enabled": bool(guidance is not None and config.enable_family_saturation),
                "filtered_candidates": saturated_family_filtered_candidate_count,
                "fallback_candidates": saturated_family_filter_fallback_count,
            },
            "quality_oracles": quality_oracle_counts,
            "stage_profile": _finalize_stage_profile_summary(stage_profile_totals, cases=executed),
            "seed": seed,
            "next_seed": next_seed,
            "guidance": {
                "strategy": config.guidance_strategy,
                "candidate_pool": candidate_pool,
                "targets": config.guidance_targets,
            },
            "effective_generator_profile": effective_generator_profile,
            "backends": backends,
            "targets": target_specs,
            "common_capabilities": list(targets.common_capabilities),
            "target_context": targets.to_dict(),
            "config": config_payload,
            "environment": environment,
            "run_provenance": run_provenance,
            "log_level": config.log_level,
            "updated_at": utc_now(),
        }
        if persist_closed_loop_state:
            closed_loop_state_payload = _build_closed_loop_state(
                seen=seen,
                signal_seen=signal_seen,
                feedback=feedback,
                guidance=guidance,
            )
            out["closed_loop_state_file"] = (
                str(persisted_closed_loop_state_path) if persisted_closed_loop_state_path is not None else ""
            )
            out["closed_loop_state_summary"] = _closed_loop_state_summary(closed_loop_state_payload)
        return out

    def write_checkpoint(status: str) -> None:
        run_writer.flush()
        if case_writer is not None:
            case_writer.flush()
        if checkpoint_file is not None:
            if persist_closed_loop_state and persisted_closed_loop_state_path is not None:
                dump_json(
                    _build_closed_loop_state(
                        seen=seen,
                        signal_seen=signal_seen,
                        feedback=feedback,
                        guidance=guidance,
                    ),
                    persisted_closed_loop_state_path,
                    compact=True,
                )
            dump_json(snapshot(status), checkpoint_file, compact=True)

    run_writer_context = JsonlWriter(run_file, compresslevel=1, buffer_lines=RUN_LOG_BUFFER_LINES)
    run_writer = run_writer_context.__enter__()
    case_writer_context = (
        JsonlWriter(
            resolved_case_log_file,
            compresslevel=1,
            buffer_lines=CASE_LOG_BUFFER_LINES,
        )
        if resolved_case_log_file
        else None
    )
    case_writer = case_writer_context.__enter__() if case_writer_context is not None else None
    include_online_weight_snapshot = config.log_level == "full" or case_writer is not None

    while True:
        if cases is not None and executed >= cases:
            break
        if duration_s is not None and executed > 0 and (time.perf_counter() - started) >= duration_s:
            break
        candidate_seed_start = next_seed
        candidate_seed_cursor = candidate_seed_start
        generate_mutate_started = time.perf_counter()
        candidates: list[Case] = []
        candidate_meta: dict[int, dict[str, Any]] = {}
        for offset in range(candidate_pool):
            skipped_replay_candidates = 0
            replay_skip_reason = ""
            last_replay_skip_reason = ""
            replay_fallback_used = False
            skipped_saturated_family_candidates = 0
            last_saturated_family_skip_reason = ""
            saturated_family_fallback_used = False
            while True:
                case_seed = candidate_seed_cursor
                candidate_seed_cursor += 1
                generated = generate_case(
                    case_seed,
                    type_aware=config.enable_type_aware_generation,
                    profile=effective_generator_profile,
                )
                if feedback is not None:
                    feedback_selector = getattr(feedback, "select_case", None) or getattr(feedback, "choose_case")
                    selected = feedback_selector(case_seed, generated)
                else:
                    selected = generated
                source = getattr(feedback, "last_candidate_source", "generated") if feedback is not None else "generated"
                metadata = (
                    getattr(feedback, "last_candidate_metadata", None)
                    if feedback is not None
                    else None
                ) or selected.metadata or _generated_candidate_metadata(generated)
                preflight = preflight_case(
                    selected,
                    enable_validation=config.enable_preflight_validation,
                    enable_repair=config.enable_preflight_repair,
                )
                candidate = preflight.case
                replay_skip_reason = _known_replay_source_filter_reason(candidate, config)
                if not replay_skip_reason:
                    saturated_roots = (
                        guidance.predicted_saturated_family_roots_for_candidate(
                            candidate,
                            include_known_families=False,
                        )
                        if guidance is not None
                        else []
                    )
                    if not saturated_roots:
                        break
                    last_saturated_family_skip_reason = ",".join(saturated_roots)
                    saturated_family_filtered_candidate_count += 1
                    skipped_saturated_family_candidates += 1
                    if skipped_saturated_family_candidates <= SATURATED_FAMILY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE:
                        continue
                    saturated_family_fallback_used = True
                    saturated_family_filter_fallback_count += 1
                    case_seed = candidate_seed_cursor
                    candidate_seed_cursor += 1
                    generated = generate_case(
                        case_seed,
                        type_aware=config.enable_type_aware_generation,
                        profile="common",
                    )
                    selected = generated
                    source = "generated_saturation_fallback"
                    metadata = _generated_candidate_metadata(generated)
                    preflight = preflight_case(
                        selected,
                        enable_validation=config.enable_preflight_validation,
                        enable_repair=config.enable_preflight_repair,
                    )
                    candidate = preflight.case
                    replay_skip_reason = _known_replay_source_filter_reason(candidate, config)
                    if replay_skip_reason:
                        replay_filtered_candidate_count += 1
                        raise RuntimeError(f"saturation fallback generated replay candidate: {replay_skip_reason}")
                    fallback_roots = (
                        guidance.predicted_saturated_family_roots_for_candidate(
                            candidate,
                            include_known_families=False,
                        )
                        if guidance is not None
                        else []
                    )
                    if fallback_roots:
                        last_saturated_family_skip_reason = ",".join(fallback_roots)
                    break
                last_replay_skip_reason = replay_skip_reason
                replay_filtered_candidate_count += 1
                skipped_replay_candidates += 1
                if skipped_replay_candidates <= REPLAY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE:
                    continue
                replay_fallback_used = True
                replay_filter_fallback_count += 1
                case_seed = candidate_seed_cursor
                candidate_seed_cursor += 1
                generated = generate_case(
                    case_seed,
                    type_aware=config.enable_type_aware_generation,
                    profile="common",
                )
                selected = generated
                source = "generated_fresh_fallback"
                metadata = _generated_candidate_metadata(generated)
                preflight = preflight_case(
                    selected,
                    enable_validation=config.enable_preflight_validation,
                    enable_repair=config.enable_preflight_repair,
                )
                candidate = preflight.case
                replay_skip_reason = _known_replay_source_filter_reason(candidate, config)
                if replay_skip_reason:
                    replay_filtered_candidate_count += 1
                    raise RuntimeError(f"fresh fallback generated replay candidate: {replay_skip_reason}")
                break
            candidate_meta[id(candidate)] = {
                "source": source,
                "generated_seed": case_seed,
                "seed_lineage": metadata.get("seed_lineage", {}),
                "mutation": metadata.get("mutation", {}),
                "feedback_selection": metadata.get("feedback_selection", metadata.get("feedback_decision", {})),
                "feedback_decision": metadata.get("feedback_decision", metadata.get("feedback_selection", {})),
                "preflight": preflight.to_dict(),
                "replay_filter": {
                    "enabled": not config.enable_replay_bug,
                    "filtered_before_candidate": skipped_replay_candidates,
                    "fallback_used": replay_fallback_used,
                    "last_skip_reason": last_replay_skip_reason,
                },
                "family_saturation_filter": {
                    "enabled": bool(guidance is not None and config.enable_family_saturation),
                    "filtered_before_candidate": skipped_saturated_family_candidates,
                    "fallback_used": saturated_family_fallback_used,
                    "last_skip_reason": last_saturated_family_skip_reason,
                },
            }
            candidates.append(candidate)
        generate_mutate_elapsed_ms = (time.perf_counter() - generate_mutate_started) * 1000
        next_seed = candidate_seed_cursor
        scheduler_feedback_elapsed_ms = 0.0
        if guidance is not None:
            guidance_started = time.perf_counter()
            guidance_selector = getattr(guidance, "select_case", None) or getattr(guidance, "choose_case")
            decision = guidance_selector(
                candidates,
                include_online_weight_snapshot=include_online_weight_snapshot,
            )
            case = decision.case
            guidance_row = decision.to_dict()
            guidance_row["strategy"] = config.guidance_strategy
            selected_operation_combo = (
                decision.analysis.operation_combo
                if getattr(decision, "analysis", None) is not None
                else None
            )
            scheduler_feedback_elapsed_ms += (time.perf_counter() - guidance_started) * 1000
        else:
            case = candidates[0]
            guidance_row = {
                "strategy": config.guidance_strategy,
                "score": 0.0,
                "features": [],
                "matched_targets": [],
                "candidate_count": 1,
            }
            selected_operation_combo = None
        selected_meta = candidate_meta.get(id(case))
        if selected_meta is None:
            selected_meta = _selected_candidate_metadata(
                case,
                config,
                bool(guidance is not None and config.enable_family_saturation),
            )
        selected_meta["operation_combo"] = selected_operation_combo or describe_operation_combo(case.program.operations)
        preflight_row = selected_meta["preflight"]
        case_seed = case.seed
        if case_writer is not None:
            case_writer.write(
                {
                    "run_id": run_id,
                    "case_index": executed,
                    "seed": case_seed,
                    "candidate_seed_start": candidate_seed_start,
                    "candidate_pool_size": candidate_pool,
                    "guidance": guidance_row,
                    "candidate_source": selected_meta["source"],
                    "seed_lineage": selected_meta["seed_lineage"],
                    "mutation": selected_meta["mutation"],
                    "feedback_selection": selected_meta.get(
                        "feedback_selection",
                        selected_meta.get("feedback_decision", {}),
                    ),
                    "feedback_decision": selected_meta.get(
                        "feedback_decision",
                        selected_meta.get("feedback_selection", {}),
                    ),
                    "operation_combo": selected_meta["operation_combo"],
                    "preflight": preflight_row,
                    "replay_filter": selected_meta["replay_filter"],
                    "family_saturation_filter": selected_meta["family_saturation_filter"],
                    "generated_at": utc_now(),
                    "case": case.to_dict(),
                }
            )
        save_artifact_for_case = _artifact_budget_available(config, artifact_saved_count)
        row = run_loaded_case(
            case,
            backends=backends,
            config=config,
            save_artifact=not config.enable_reducer and save_artifact_for_case,
            backend_instances=backend_instances,
            environment=environment,
            target_specs=target_specs,
            config_payload=config_payload,
        )
        row_stage_profile = _stage_profile_with_total(row.get("stage_profile", {}))
        row_stage_profile["generate_mutate_ms"] += generate_mutate_elapsed_ms
        countable_row_findings = _countable_row_findings(row)
        if countable_row_findings and config.enable_reducer:
            from datadiff.reducer import reduce_case

            reducer_started = time.perf_counter()
            reduced = reduce_case(
                case,
                backends=backends,
                config=ExperimentConfig(
                    enable_type_aware_generation=config.enable_type_aware_generation,
                    enable_normalizer=config.enable_normalizer,
                    enable_differential_oracle=config.enable_differential_oracle,
                    enable_metamorphic_oracle=config.enable_metamorphic_oracle,
                    enable_feedback=False,
                    enable_reducer=False,
                    enable_artifact=False,
                    oracle_mode=config.oracle_mode,
                    generator_profile=config.generator_profile,
                    metamorphic_variant_limit=config.metamorphic_variant_limit,
                ),
                target_kinds=[finding["kind"] for finding in countable_row_findings],
                target_roots=[finding.get("root_cause", "unknown") for finding in countable_row_findings],
            )
            scheduler_feedback_elapsed_ms += (time.perf_counter() - reducer_started) * 1000
            reduced_row = run_loaded_case(
                reduced,
                backends=backends,
                config=config,
                save_artifact=_artifact_budget_available(config, artifact_saved_count),
                backend_instances=backend_instances,
                environment=environment,
                target_specs=target_specs,
                config_payload=config_payload,
            )
            reduced_row["original_case"] = case.to_dict()
            reduced_row["reduction"] = {
                "original_rows": len(case.tables[0].rows),
                "reduced_rows": len(reduced.tables[0].rows),
                "original_ops": len(case.program.operations),
                "reduced_ops": len(reduced.program.operations),
            }
            row = reduced_row
            countable_row_findings = _countable_row_findings(row)
            row_stage_profile = _stage_profile_with_total(row.get("stage_profile", {}))
            row_stage_profile["generate_mutate_ms"] += generate_mutate_elapsed_ms
        if countable_row_findings and row.get("bug_dir"):
            artifact_saved_count += 1
            row["artifact_saved"] = True
        elif countable_row_findings and config.enable_artifact:
            row["artifact_saved"] = False
            row["artifact_skipped_reason"] = "artifact_limit_reached"

        sig = row["behavior_signature"]
        discovery_sig = str(row.get("discovery_signature", sig))
        row["is_new_behavior"] = discovery_sig not in seen
        seen.add(discovery_sig)
        signal_sig = signal_signature(row)
        row["signal_signature"] = signal_sig
        row["signal_new_behavior"] = bool(row["is_new_behavior"]) and signal_sig not in signal_seen
        if row["is_new_behavior"]:
            signal_seen.add(signal_sig)
        row["candidate_source"] = selected_meta["source"]
        row["seed_lineage"] = selected_meta["seed_lineage"]
        row["mutation"] = selected_meta["mutation"]
        row["feedback_selection"] = selected_meta.get("feedback_selection", selected_meta.get("feedback_decision", {}))
        row["feedback_decision"] = selected_meta.get("feedback_decision", selected_meta.get("feedback_selection", {}))
        row["operation_combo"] = selected_meta["operation_combo"]
        row["preflight"] = preflight_row
        row["replay_filter"] = selected_meta["replay_filter"]
        row["family_saturation_filter"] = selected_meta["family_saturation_filter"]
        row["candidate_seed_start"] = candidate_seed_start
        row["candidate_pool_size"] = candidate_pool
        row["case_index"] = executed
        row["elapsed_s"] = round(time.perf_counter() - started, 6)
        quality_oracles = evaluate_quality_oracles(
            case,
            row,
            candidate_source=selected_meta["source"],
            preflight=preflight_row,
            guidance_decision=guidance_row,
            guidance_strategy=config.guidance_strategy,
            guidance_targets=guidance_targets,
        )
        row["quality_oracles"] = [oracle.to_dict() for oracle in quality_oracles]
        for oracle in row["quality_oracles"]:
            quality_oracle_counts[f"{oracle['name']}:{oracle['verdict']}"] = (
                quality_oracle_counts.get(f"{oracle['name']}:{oracle['verdict']}", 0) + 1
            )
        row_findings = row.get("findings") or []
        finding_outcomes = (
            analyze_finding_outcomes(
                row_findings,
                known_saturated_bug_families=config.known_saturated_bug_families,
            )
            if row_findings
            else None
        )
        if feedback is not None:
            known_families = config.known_saturated_bug_families
            reward_signals = row_reward_signals(
                row,
                known_saturated_bug_families=known_families,
                finding_outcomes=finding_outcomes,
            )
            row_candidate_families = list(finding_outcomes.candidate_bug_families) if finding_outcomes else []
            row_candidate_signatures = list(finding_outcomes.candidate_bug_signatures) if finding_outcomes else []
            rewardable_semantic_divergence = bool(reward_signals["rewardable_semantic_divergence"])
            resolved_semantic_only = (
                bool(reward_signals["resolved_semantic_divergence_count"])
                and not bool(reward_signals["candidate_bug"])
                and not rewardable_semantic_divergence
            )
            feedback_finding = bool(row_candidate_families) or rewardable_semantic_divergence
            feedback_eligible, feedback_skip_reason = _feedback_storage_decision(
                case,
                candidate_source=selected_meta["source"],
                seed_lineage=selected_meta["seed_lineage"],
            )
            if feedback_eligible and resolved_semantic_only:
                feedback_eligible = False
                feedback_skip_reason = "resolved_semantic_divergence"
            row["feedback_eligible"] = feedback_eligible
            row["feedback_skip_reason"] = feedback_skip_reason
            feedback_summary = feedback_summary_for_case(
                row,
                known_saturated_bug_families=known_families,
                finding_outcomes=finding_outcomes,
            )
            if feedback_eligible:
                feedback_started = time.perf_counter()
                row_target_keys = _feedback_target_keys(guidance_row, selected_meta["operation_combo"])
                row["stored_in_feedback_corpus"] = feedback.record(
                    case,
                    sig,
                    feedback_finding,
                    novelty_signature=signal_sig,
                    discovery_signature=discovery_sig,
                    candidate_bug_families=row_candidate_families,
                    target_keys=row_target_keys,
                    schedule_delta=float(feedback_summary.get("seed_schedule_delta", 0.0) or 0.0),
                )
                scheduler_feedback_elapsed_ms += (time.perf_counter() - feedback_started) * 1000
                row["feedback_corpus_persisted"] = feedback.last_persisted_to_disk
                row["feedback_record_skip_reason"] = feedback.last_record_skip_reason
            else:
                feedback.last_persisted_to_disk = False
                row["stored_in_feedback_corpus"] = False
                row["feedback_corpus_persisted"] = False
                row["feedback_record_skip_reason"] = ""
            feedback_summary["stored_in_feedback_corpus"] = bool(row["stored_in_feedback_corpus"])
            feedback_summary["source_reward_adjustment"] = source_reward_adjustment_from_summary(
                feedback_summary,
                candidate_source=selected_meta["source"],
            )
            row["feedback_summary"] = feedback_summary
            source_reward_started = time.perf_counter()
            feedback_outcome_recorder = getattr(feedback, "record_candidate_outcome", None) or getattr(
                feedback,
                "record_candidate_result",
            )
            row["source_reward"] = feedback_outcome_recorder(
                selected_meta["source"],
                has_finding=feedback_finding,
                is_new_behavior=bool(row["signal_new_behavior"]) and not resolved_semantic_only,
                preflight=preflight_row,
                candidate_bug=bool(reward_signals["candidate_bug"]),
                semantic_divergence=rewardable_semantic_divergence,
                false_positive=bool(reward_signals["false_positive"]),
                candidate_bug_families=row_candidate_families,
                candidate_bug_signatures=row_candidate_signatures,
                reward_adjustment=float(feedback_summary.get("source_reward_adjustment", 0.0) or 0.0),
            )
            scheduler_feedback_elapsed_ms += (time.perf_counter() - source_reward_started) * 1000
            row["source_scheduler"] = _source_scheduler_snapshot(feedback)
        else:
            row["stored_in_feedback_corpus"] = False
            row["feedback_corpus_persisted"] = False
            row["feedback_eligible"] = False
            row["feedback_skip_reason"] = "feedback_disabled"
            row["feedback_record_skip_reason"] = ""
            row["source_reward"] = None
            row["source_scheduler"] = []
            row["feedback_summary"] = feedback_summary_for_case(
                row,
                known_saturated_bug_families=config.known_saturated_bug_families,
                finding_outcomes=finding_outcomes,
            )
        if guidance is not None:
            guidance_record_started = time.perf_counter()
            guidance.record_result(case, row, finding_outcomes=finding_outcomes)
            scheduler_feedback_elapsed_ms += (time.perf_counter() - guidance_record_started) * 1000
        row["guidance"] = guidance_row
        row_stage_profile["scheduler_feedback_ms"] += scheduler_feedback_elapsed_ms
        logging_started = time.perf_counter()
        preflight_repaired_count += int(bool(preflight_row.get("repaired", False)))
        preflight_fallback_count += int(bool(preflight_row.get("fallback_used", False)))
        preflight_invalid_count += int(not bool(preflight_row.get("valid", True)))
        findings_count += len(row["findings"])
        new_behavior_count += int(row["is_new_behavior"])
        signal_new_behavior_count += int(row["signal_new_behavior"])
        row_stage_profile["logging_artifact_ms"] += (time.perf_counter() - logging_started) * 1000
        row_stage_profile["total_case_wall_ms"] = sum(
            row_stage_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row["stage_profile"] = row_stage_profile
        row["duration_ms"] = row_stage_profile["total_case_wall_ms"]
        run_writer.write(_compact_log_row(row, config.log_level))
        stage_profile_totals = _merge_stage_profile(stage_profile_totals, row_stage_profile)
        executed += 1

        now = time.perf_counter()
        if checkpoint_interval_s is not None and (now - last_checkpoint) >= checkpoint_interval_s:
            write_checkpoint("running")
            last_checkpoint = now
        if (
            progress_callback is not None
            and progress_interval_s is not None
            and (now - last_progress) >= progress_interval_s
        ):
            run_writer.flush()
            if case_writer is not None:
                case_writer.flush()
            progress_callback(snapshot("running"))
            last_progress = now

        if duration_s is not None and (time.perf_counter() - started) >= duration_s:
            break

    if case_writer_context is not None:
        case_writer_context.__exit__(None, None, None)
    run_writer_context.__exit__(None, None, None)

    elapsed_s = time.perf_counter() - started
    meta = snapshot("completed")
    meta["elapsed_s"] = elapsed_s
    meta["throughput_cases_s"] = executed / elapsed_s if elapsed_s else 0.0
    if persist_closed_loop_state and persisted_closed_loop_state_path is not None:
        dump_json(
            _build_closed_loop_state(
                seen=seen,
                signal_seen=signal_seen,
                feedback=feedback,
                guidance=guidance,
            ),
            persisted_closed_loop_state_path,
            compact=True,
        )
    dump_json(meta, run_meta_path(run_file), compact=True)
    write_checkpoint("completed")
    if progress_callback is not None:
        progress_callback(meta)
    return run_file


def _countable_row_findings(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        finding
        for finding in row.get("findings", []) or []
        if _is_countable_finding_dict(finding)
    ]


def _countable_finding_objects(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if _is_countable_finding_dict(finding.to_dict())]


def _is_countable_finding_dict(finding: dict[str, Any]) -> bool:
    return counts_as_bug_evidence(finding)


def _artifact_budget_available(config: ExperimentConfig, saved_count: int) -> bool:
    if not config.enable_artifact:
        return False
    if config.artifact_limit is None:
        return True
    return saved_count < max(0, int(config.artifact_limit))
