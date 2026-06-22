from __future__ import annotations

from typing import Any

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


def _case_log_row(
    *,
    run_id: str,
    case_index: int,
    case_seed: int,
    candidate_seed_start: int,
    candidate_pool: int,
    guidance_row: dict[str, Any],
    selected_meta: dict[str, Any],
    backend_pair_pool: tuple[str, ...] | list[str],
    preflight_row: dict[str, Any],
    generated_at: str,
    case: Any,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_index": case_index,
        "seed": case_seed,
        "candidate_seed_start": candidate_seed_start,
        "candidate_pool_size": candidate_pool,
        "guidance": guidance_row,
        "candidate_source": selected_meta["source"],
        "seed_lineage": selected_meta["seed_lineage"],
        "mutation": selected_meta["mutation"],
        "feedback_decision": selected_meta.get("feedback_decision", {}),
        "quality_archive_context": selected_meta.get("quality_archive_context", {}),
        "generator_profile_selection": selected_meta.get("generator_profile_selection", {}),
        "semantic_objective_selection": selected_meta.get("semantic_objective_selection", {}),
        "metamorphic_relation_selection": selected_meta.get("metamorphic_relation_selection", {}),
        "version_pair_selection": selected_meta.get("version_pair_selection", {}),
        "selected_version_pair": selected_meta.get("selected_version_pair", ""),
        "backend_pair_pool": list(backend_pair_pool),
        "case_learning_context": selected_meta.get("case_learning_context", []),
        "operation_combo": selected_meta["operation_combo"],
        "preflight": preflight_row,
        "replay_filter": selected_meta["replay_filter"],
        "family_saturation_filter": selected_meta["family_saturation_filter"],
        "generated_at": generated_at,
        "case": case.to_dict(),
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
    adaptive_learning = feedback_state.get("adaptive_learning", {}) if isinstance(feedback_state, dict) else {}
    quality_archive = feedback_state.get("quality_archive", {}) if isinstance(feedback_state, dict) else {}
    seed_quota = feedback_state.get("seed_quota", {}) if isinstance(feedback_state, dict) else {}
    return {
        "seen_signature_count": len(state.get("seen_signatures", []) or []),
        "signal_seen_signature_count": len(state.get("signal_seen_signatures", []) or []),
        "feedback_interesting_case_count": len(feedback_state.get("interesting_cases", []) or []),
        "feedback_stored_candidate_family_count": len(feedback_state.get("stored_candidate_bug_families", {}) or {}),
        "feedback_stored_target_key_count": len(feedback_state.get("stored_target_keys", {}) or {}),
        "adaptive_learning_health": _adaptive_learning_health_summary(adaptive_learning),
        "quality_archive_health": _quality_archive_health_summary(quality_archive),
        "seed_quota_health": _seed_quota_health_summary(feedback_state, seed_quota),
        "champion_corpus_health": _champion_corpus_health_summary(feedback_state),
        "guidance_feature_count": len(guidance_state.get("feature_counts", {}) or {}),
        "guidance_frontier_bucket_count": len(guidance_state.get("frontier_bucket_counts", {}) or {}),
        "guidance_candidate_bug_family_count": len(guidance_state.get("candidate_bug_family_counts", {}) or {}),
    }


def _seed_quota_health_summary(feedback_state: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(feedback_state, dict):
        return {}
    cluster_counts = feedback_state.get("stored_cluster_keys", {})
    cluster_count = len(cluster_counts) if isinstance(cluster_counts, dict) else 0
    seed_count = len(feedback_state.get("interesting_cases", []) or [])
    return {
        "enabled": bool(feedback_state.get("enable_seed_quota", bool(state.get("enabled", True)))),
        "cluster_count": cluster_count,
        "seed_count": seed_count,
        "active": bool(seed_count > 0 and cluster_count > 0),
        "min_quota_per_active_cell": int(state.get("min_quota_per_active_cell", 1) or 1)
        if isinstance(state, dict)
        else 1,
    }


def _champion_corpus_health_summary(feedback_state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(feedback_state, dict):
        return {}
    family_hits = feedback_state.get("champion_family_hits", {})
    promoted = feedback_state.get("champion_promoted_families", [])
    return {
        "enabled": bool(feedback_state.get("enable_champion_corpus", True)),
        "donor_bandit_enabled": bool(feedback_state.get("enable_champion_graft_donor_bandit", True)),
        "family_hit_count": len(family_hits) if isinstance(family_hits, dict) else 0,
        "promoted_family_count": len(promoted) if isinstance(promoted, list) else 0,
        "version_id": str(feedback_state.get("champion_version_id", "") or ""),
    }


def _adaptive_learning_health_summary(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict) or str(state.get("schema_version", "") or "") != "adaptive-learning-v1":
        return {}
    bandits = state.get("bandits", {}) if isinstance(state.get("bandits", {}), dict) else {}
    arm_count = 0
    total_pulls = 0
    reward_model_update_count = 0
    reward_model_feature_count = 0
    runtime_cost_observation_count = 0
    runtime_cost_total = 0.0
    scope_pull_counts: dict[str, int] = {}
    scope_arm_counts: dict[str, int] = {}
    health_penalties: list[float] = []
    uncertainties: list[float] = []
    for scope, bandit in bandits.items():
        if not isinstance(bandit, dict):
            continue
        scope_key = str(scope or "")
        scope_arm_count = 0
        scope_pull_count = 0
        for arm in bandit.get("arms", []) or []:
            if not isinstance(arm, dict):
                continue
            arm_count += 1
            scope_arm_count += 1
            arm_pulls = int(arm.get("pulls", 0) or 0)
            total_pulls += arm_pulls
            scope_pull_count += arm_pulls
            pulls = max(1, arm_pulls)
            arm_runtime_cost = float(arm.get("runtime_cost_total", 0.0) or 0.0)
            runtime_cost_total += arm_runtime_cost
            if arm_pulls > 0:
                runtime_cost_observation_count += arm_pulls
            runtime_cost = arm_runtime_cost / pulls
            false_positive = float(arm.get("false_positive_count", 0) or 0) / pulls
            invalid = float(arm.get("invalid_count", 0) or 0) / pulls
            health_penalties.append(min(1.5, 0.35 * runtime_cost + 0.60 * false_positive + 0.25 * invalid))
        reward_model = bandit.get("reward_model", {}) if isinstance(bandit.get("reward_model", {}), dict) else {}
        reward_model_update_count += int(reward_model.get("total_updates", 0) or 0)
        feature_counts = reward_model.get("feature_counts", {})
        if isinstance(feature_counts, dict):
            reward_model_feature_count += len(feature_counts)
            uncertainties.extend(1.0 / ((1.0 + float(count or 0.0)) ** 0.5) for count in feature_counts.values())
        if scope_key:
            scope_pull_counts[scope_key] = max(scope_pull_count, int(bandit.get("total_pulls", 0) or 0))
            scope_arm_counts[scope_key] = scope_arm_count
    version_memory = state.get("version_memory", {})
    version_memory_key_count = 0
    if isinstance(version_memory, dict):
        reward_counts = version_memory.get("reward_counts", {})
        version_memory_key_count = len(reward_counts) if isinstance(reward_counts, dict) else 0
    continual_memory = state.get("continual_priority_memory", {})
    continual_summary = {}
    if isinstance(continual_memory, dict):
        family_priorities = continual_memory.get("family_priorities", {})
        feature_counts = continual_memory.get("feature_counts", {})
        continual_summary = {
            "imported_ledger_count": int(continual_memory.get("imported_ledger_count", 0) or 0),
            "imported_family_count": int(continual_memory.get("imported_family_count", 0) or 0),
            "imported_health_feedback_count": int(
                continual_memory.get("imported_health_feedback_count", 0) or 0
            ),
            "family_count": len(family_priorities) if isinstance(family_priorities, dict) else 0,
            "feature_count": len(feature_counts) if isinstance(feature_counts, dict) else 0,
            "health_feedback_feature_count": len(
                continual_memory.get("feature_health_counts", {})
                if isinstance(continual_memory.get("feature_health_counts", {}), dict)
                else {}
            ),
        }
    exploration_memory = state.get("exploration_memory", {})
    exploration_summary = {}
    if isinstance(exploration_memory, dict):
        context_counts = exploration_memory.get("context_counts", {})
        action_counts = exploration_memory.get("action_counts", {})
        exploration_summary = {
            "total_records": int(exploration_memory.get("total_records", 0) or 0),
            "context_count": len(context_counts) if isinstance(context_counts, dict) else 0,
            "action_count": len(action_counts) if isinstance(action_counts, dict) else 0,
        }
    return {
        "schema_version": "adaptive-learning-health-v1",
        "bandit_count": len(bandits),
        "arm_count": arm_count,
        "total_pulls": total_pulls,
        "reward_model_update_count": reward_model_update_count,
        "reward_model_feature_count": reward_model_feature_count,
        "version_memory_key_count": version_memory_key_count,
        "runtime_cost_observation_count": runtime_cost_observation_count,
        "runtime_cost_total": runtime_cost_total,
        "scope_pull_counts": dict(sorted(scope_pull_counts.items())),
        "scope_arm_counts": dict(sorted(scope_arm_counts.items())),
        "value_catalog_entry_pulls": int(scope_pull_counts.get("value_catalog_entry", 0) or 0),
        "value_catalog_entry_arm_count": int(scope_arm_counts.get("value_catalog_entry", 0) or 0),
        "bd_axis_weight_pulls": int(scope_pull_counts.get("bd_axis_weights", 0) or 0),
        "bd_axis_weight_arm_count": int(scope_arm_counts.get("bd_axis_weights", 0) or 0),
        "seed_energy_tier_pulls": int(scope_pull_counts.get("seed_energy_tier", 0) or 0),
        "seed_energy_tier_arm_count": int(scope_arm_counts.get("seed_energy_tier", 0) or 0),
        "champion_graft_donor_pulls": int(scope_pull_counts.get("champion_graft_donor", 0) or 0),
        "champion_graft_donor_arm_count": int(scope_arm_counts.get("champion_graft_donor", 0) or 0),
        "continual_priority_memory": continual_summary,
        "avg_health_penalty": sum(health_penalties) / len(health_penalties) if health_penalties else 0.0,
        "max_health_penalty": max(health_penalties) if health_penalties else 0.0,
        "avg_uncertainty": sum(uncertainties) / len(uncertainties) if uncertainties else 0.0,
        "exploration_memory": exploration_summary,
    }


def _quality_archive_health_summary(state: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state, dict) or str(state.get("schema_version", "") or "") not in {
        "quality-diversity-archive-v1",
        "quality-diversity-archive-v2",
    }:
        return {}
    cells = [cell for cell in state.get("cells", []) or [] if isinstance(cell, dict)]
    seed_count = 0
    elite_seed_count = 0
    child_cell_count = 0
    split_cell_count = 0
    reward_count = 0
    outcome_count = 0
    invalid_count = 0
    fallback_count = 0
    false_positive_count = 0
    for cell in cells:
        seeds = [seed for seed in cell.get("seeds", []) or [] if isinstance(seed, dict)]
        max_elites = max(0, int(cell.get("max_elites", state.get("max_elites_per_cluster", 4)) or 0))
        children = [child for child in cell.get("children", []) or [] if isinstance(child, dict)]
        child_cell_count += len(children)
        if str(cell.get("split_axis", "") or ""):
            split_cell_count += 1
        seed_count += len(seeds)
        elite_seed_count += min(len(seeds), max_elites)
        reward_count += int(cell.get("reward_count", 0) or 0)
        outcome_count += int(cell.get("outcome_count", 0) or 0)
        invalid_count += int(cell.get("invalid_count", 0) or 0)
        fallback_count += int(cell.get("fallback_count", 0) or 0)
        false_positive_count += int(cell.get("false_positive_count", 0) or 0)
    return {
        "schema_version": "quality-archive-health-v1",
        "cell_count": len(cells),
        "hierarchical_enabled": bool(state.get("enable_hierarchical", False)),
        "child_cell_count": child_cell_count,
        "split_cell_count": split_cell_count,
        "seed_count": seed_count,
        "elite_seed_count": elite_seed_count,
        "reward_count": reward_count,
        "outcome_count": outcome_count,
        "invalid_count": invalid_count,
        "fallback_count": fallback_count,
        "false_positive_count": false_positive_count,
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
        "execution_profile": row.get("execution_profile", {}),
        "fuzz_iteration": row.get("fuzz_iteration", {}),
        "findings": row.get("findings", []),
        "bug_dir": row.get("bug_dir", ""),
        "candidate_source": row.get("candidate_source", "generated"),
        "seed_lineage": row.get("seed_lineage", {}),
        "mutation": row.get("mutation", {}),
        "feedback_decision": row.get("feedback_decision", {}),
        "quality_archive_context": row.get("quality_archive_context", {}),
        "generator_profile_selection": row.get("generator_profile_selection", {}),
        "selected_generator_profile": row.get("selected_generator_profile", ""),
        "semantic_objective_selection": row.get("semantic_objective_selection", {}),
        "selected_semantic_objective": row.get("selected_semantic_objective", ""),
        "metamorphic_relation_selection": row.get("metamorphic_relation_selection", {}),
        "selected_metamorphic_relation": row.get("selected_metamorphic_relation", ""),
        "version_pair_selection": row.get("version_pair_selection", {}),
        "selected_version_pair": row.get("selected_version_pair", ""),
        "backend_pair_selection": row.get("backend_pair_selection", {}),
        "backend_pair_priority": row.get("backend_pair_priority", []),
        "backend_pair_feedback": row.get("backend_pair_feedback", {}),
        "case_learning_context": row.get("case_learning_context", []),
        "disagreement_descriptor": row.get("disagreement_descriptor", {}),
        "case_fingerprint": row.get("case_fingerprint", {}),
        "oracle_cross_validation": row.get("oracle_cross_validation", {}),
        "metamorphic_selection": row.get("metamorphic_selection", {}),
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
