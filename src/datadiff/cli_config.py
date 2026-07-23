from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES, ExperimentConfig
from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    merge_exploration_objective_rules,
)
from datadiff.guidance import parse_guidance_targets
from datadiff.method_arms import DEFAULT_METHOD_ARM_ID
from datadiff.util import load_json


ADAPTIVE_COMPONENTS = (
    "scheduler_learning",
    "profile_learning",
    "semantic_objective_learning",
    "metamorphic_relation_learning",
    "version_pair_learning",
    "backend_pair_learning",
    "profile_capability_filter",
    "mutation_operator_learning",
    "operator_swarm",
    "ir_rewrite_mutations",
    "divergence_conditioned",
    "shrink_mutations",
    "value_catalog",
    "quality_archive",
    "hierarchical_archive",
    "bd_axis_bandit",
    "bayesian_exploration",
    "seed_quota",
    "seed_energy_batch",
    "seed_energy_tier",
    "per_operator_energy",
    "lineage_rarity",
    "minhash_dedup",
    "disagreement_bd_axis",
    "lhs_seeding",
    "champion_corpus",
    "champion_graft_donor",
    "local_source_scheduler",
    "runtime_cost_learning",
    "cost_normalized_reward",
    "active_learning",
    "online_reward_model",
    "continual_learning",
    "scheduler_annealing",
)

ADAPTIVE_COMPONENT_ALIASES = {
    "scheduler": "scheduler_learning",
    "bandit": "scheduler_learning",
    "contextual_bandit": "scheduler_learning",
    "generator_profile_learning": "profile_learning",
    "semantic_objective": "semantic_objective_learning",
    "semantic_objective_learning": "semantic_objective_learning",
    "objective_learning": "semantic_objective_learning",
    "metamorphic_relation": "metamorphic_relation_learning",
    "metamorphic_relation_learning": "metamorphic_relation_learning",
    "mr_learning": "metamorphic_relation_learning",
    "mr_type_learning": "metamorphic_relation_learning",
    "version_pair": "version_pair_learning",
    "version_pair_learning": "version_pair_learning",
    "cross_version_pair_learning": "version_pair_learning",
    "backend_pair": "backend_pair_learning",
    "backend_pair_learning": "backend_pair_learning",
    "backend_pair_bandit": "backend_pair_learning",
    "differential_pair_learning": "backend_pair_learning",
    "profile_filter": "profile_capability_filter",
    "capability_filter": "profile_capability_filter",
    "mutation_learning": "mutation_operator_learning",
    "operator_learning": "mutation_operator_learning",
    "operator_swarm": "operator_swarm",
    "mutation_operator_swarm": "operator_swarm",
    "mopt": "operator_swarm",
    "mopt_swarm": "operator_swarm",
    "ir_rewrite": "ir_rewrite_mutations",
    "ir_rewrite_mutation": "ir_rewrite_mutations",
    "ir_rewrite_mutations": "ir_rewrite_mutations",
    "typed_ir_rewrite": "ir_rewrite_mutations",
    "typed_ir_rewrites": "ir_rewrite_mutations",
    "subtree_rewrite": "ir_rewrite_mutations",
    "subtree_rewrites": "ir_rewrite_mutations",
    "filter_pushdown_rewrite": "ir_rewrite_mutations",
    "divergence": "divergence_conditioned",
    "divergence_conditioned": "divergence_conditioned",
    "divergence_conditioned_mutations": "divergence_conditioned",
    "divergence_affinity": "divergence_conditioned",
    "shrink": "shrink_mutations",
    "shrink_mutation": "shrink_mutations",
    "shrink_mutations": "shrink_mutations",
    "grow_shrink": "shrink_mutations",
    "value_catalog": "value_catalog",
    "value_catalog_entry": "value_catalog",
    "adversarial_value_catalog": "value_catalog",
    "literal_catalog": "value_catalog",
    "map_elites": "quality_archive",
    "quality_diversity": "quality_archive",
    "hierarchical_archive": "hierarchical_archive",
    "hierarchical_map_elites": "hierarchical_archive",
    "adaptive_archive_granularity": "hierarchical_archive",
    "bd_axis": "bd_axis_bandit",
    "bd_axis_bandit": "bd_axis_bandit",
    "bd_axis_learning": "bd_axis_bandit",
    "bd_axis_weights": "bd_axis_bandit",
    "behavioral_axis_bandit": "bd_axis_bandit",
    "behavioral_descriptor_axis_bandit": "bd_axis_bandit",
    "bayesian_exploration": "bayesian_exploration",
    "bayesian_explorer": "bayesian_exploration",
    "good_turing": "bayesian_exploration",
    "good_turing_exploration": "bayesian_exploration",
    "discovery_rate": "bayesian_exploration",
    "discovery_rate_exploration": "bayesian_exploration",
    "adaptive_exploration": "bayesian_exploration",
    "energy_quota": "seed_quota",
    "seed_energy_quota": "seed_quota",
    "corpus_quota": "seed_quota",
    "seed_energy": "seed_energy_batch",
    "seed_energy_batch": "seed_energy_batch",
    "seed_energy_tier": "seed_energy_tier",
    "seed_energy_tier_bandit": "seed_energy_tier",
    "seed_energy_tier_scope": "seed_energy_tier",
    "seed_power": "seed_energy_batch",
    "seed_power_scheduling": "seed_energy_batch",
    "power_schedule": "seed_energy_batch",
    "power_scheduling": "seed_energy_batch",
    "afl_fast": "seed_energy_batch",
    "afl_fast_seed_energy": "seed_energy_batch",
    "per_operator_energy": "per_operator_energy",
    "operator_energy": "per_operator_energy",
    "mutation_operator_energy": "per_operator_energy",
    "operator_power": "per_operator_energy",
    "operator_power_scheduling": "per_operator_energy",
    "per_operator_power": "per_operator_energy",
    "afl_fast_operator_energy": "per_operator_energy",
    "lineage": "lineage_rarity",
    "lineage_rarity": "lineage_rarity",
    "lineage_scheduler": "lineage_rarity",
    "lineage_graph": "lineage_rarity",
    "minhash": "minhash_dedup",
    "minhash_dedup": "minhash_dedup",
    "minhash_deduplication": "minhash_dedup",
    "fingerprint_dedup": "minhash_dedup",
    "fingerprint_deduplication": "minhash_dedup",
    "disagreement_bd": "disagreement_bd_axis",
    "disagreement_bd_axis": "disagreement_bd_axis",
    "backend_disagreement_axis": "disagreement_bd_axis",
    "backend_disagreement_bd_axis": "disagreement_bd_axis",
    "lhs": "lhs_seeding",
    "lhs_seeding": "lhs_seeding",
    "latin_hypercube": "lhs_seeding",
    "latin_hypercube_seeding": "lhs_seeding",
    "champion": "champion_corpus",
    "champions": "champion_corpus",
    "champion_seed": "champion_corpus",
    "champion_seeds": "champion_corpus",
    "champion_corpus": "champion_corpus",
    "cross_version_seed": "champion_corpus",
    "champion_graft": "champion_graft_donor",
    "champion_graft_donor": "champion_graft_donor",
    "champion_graft_donor_bandit": "champion_graft_donor",
    "champion_donor": "champion_graft_donor",
    "cross_version_seeds": "champion_corpus",
    "source_scheduler": "local_source_scheduler",
    "feedback_source_scheduler": "local_source_scheduler",
    "runtime_cost": "runtime_cost_learning",
    "cost_learning": "runtime_cost_learning",
    "cost_aware_learning": "runtime_cost_learning",
    "runtime_cost_penalty": "runtime_cost_learning",
    "cost_normalized": "cost_normalized_reward",
    "cost_normalized_reward": "cost_normalized_reward",
    "cost_normalization": "cost_normalized_reward",
    "elapsed_cost_normalization": "cost_normalized_reward",
    "active": "active_learning",
    "active_learning": "active_learning",
    "uncertainty_sampling": "active_learning",
    "online_exploration": "active_learning",
    "exploration_memory": "active_learning",
    "reward_model": "online_reward_model",
    "online_reward": "online_reward_model",
    "online_reward_model": "online_reward_model",
    "statistical_reward_model": "online_reward_model",
    "continual": "continual_learning",
    "continual_learning": "continual_learning",
    "cross_version_learning": "continual_learning",
    "version_transfer": "continual_learning",
    "annealing": "scheduler_annealing",
    "scheduler_annealing": "scheduler_annealing",
    "simulated_annealing": "scheduler_annealing",
    "annealed_scheduler": "scheduler_annealing",
}


def parse_exploration_objective_rules(value: str | None) -> list[ExplorationObjectiveRule]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        payload = load_json(Path(text[1:])) if text.startswith("@") else json.loads(text)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"invalid exploration objective rule payload: {exc}"
        ) from exc
    if isinstance(payload, dict):
        if isinstance(payload.get("exploration_objective_rules"), list):
            payload = payload["exploration_objective_rules"]
        elif isinstance(payload.get("rules"), list):
            payload = payload["rules"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise argparse.ArgumentTypeError(
            "--exploration-objective-rules must be a JSON object/list or @path"
        )
    try:
        return merge_exploration_objective_rules(payload)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"invalid exploration objective rule payload: {exc}"
        ) from exc


def parse_adaptive_components(
    value: str | list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif value:
        raw_items = [str(item).strip() for item in value]
    components: set[str] = set()
    for raw in raw_items:
        item = raw.strip().lower().replace("-", "_")
        if not item or item in {"none", "off", "false", "0"}:
            continue
        resolved = ADAPTIVE_COMPONENT_ALIASES.get(item, item)
        if resolved not in ADAPTIVE_COMPONENTS:
            allowed = ",".join(ADAPTIVE_COMPONENTS)
            raise argparse.ArgumentTypeError(
                f"unknown adaptive component '{raw}'; expected one of: {allowed}"
            )
        components.add(resolved)
    return components


def parse_adaptive_component_tuple(
    value: str | list[str] | tuple[str, ...] | set[str] | None,
) -> tuple[str, ...]:
    raw_items: list[str] = []
    if isinstance(value, str):
        raw_items = [item.strip() for item in value.split(",")]
    elif value:
        for item in value:
            raw_items.extend(part.strip() for part in str(item).split(","))
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        for component in parse_adaptive_components(raw):
            if component not in seen:
                ordered.append(component)
                seen.add(component)
    return tuple(ordered)


def adaptive_component_config(disabled_components: set[str]) -> dict[str, bool]:
    return {component: component not in disabled_components for component in ADAPTIVE_COMPONENTS}


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    disabled_adaptive_components = parse_adaptive_components(
        getattr(args, "disable_adaptive_components", "")
    )
    research_control_arm = str(getattr(args, "research_control_arm", "") or "")
    return ExperimentConfig(
        method_arm=(
            research_control_arm
            or str(getattr(args, "method_arm", DEFAULT_METHOD_ARM_ID) or DEFAULT_METHOD_ARM_ID)
        ),
        enable_type_aware_generation=not args.disable_type_aware_generation,
        enable_normalizer=not args.disable_normalizer,
        enable_differential_oracle=not args.disable_differential_oracle,
        enable_metamorphic_oracle=args.enable_metamorphic_oracle,
        enable_witness_oracle=bool(getattr(args, "enable_witness_oracle", False)),
        enable_feedback=not args.disable_feedback,
        enable_replay_bug=bool(getattr(args, "enable_replay_bug", False)),
        enable_reducer=args.enable_reducer,
        enable_artifact=not args.disable_artifact,
        enable_parallel_backend_execution=(
            bool(getattr(args, "enable_parallel_backend_execution", False))
            and not bool(getattr(args, "disable_parallel_backend_execution", False))
        ),
        enable_backend_session_reuse=not bool(
            getattr(args, "disable_backend_session_reuse", False)
        ),
        enable_backend_sampling=bool(getattr(args, "enable_backend_sampling", False)),
        backend_sample_size=max(2, int(getattr(args, "backend_sample_size", None) or 3)),
        backend_full_sweep_interval=max(
            0,
            int(
                16
                if getattr(args, "backend_full_sweep_interval", None) is None
                else args.backend_full_sweep_interval
            ),
        ),
        backend_sample_confirm_candidates=(
            True
            if getattr(args, "backend_sample_confirm_candidates", None) is None
            else bool(args.backend_sample_confirm_candidates)
        ),
        backend_sampling_calibration_cases=max(
            0,
            int(
                8
                if getattr(args, "backend_sampling_calibration_cases", None) is None
                else args.backend_sampling_calibration_cases
            ),
        ),
        backend_sampling_candidate_burst_cases=max(
            0,
            int(
                8
                if getattr(args, "backend_sampling_candidate_burst_cases", None) is None
                else args.backend_sampling_candidate_burst_cases
            ),
        ),
        backend_sampling_candidate_burst_novel_only=bool(
            getattr(args, "backend_sampling_candidate_burst_novel_only", False)
        ),
        backend_sample_confirmation_recheck_count=(
            None
            if getattr(args, "backend_sample_confirmation_recheck_count", None) is None
            else max(0, int(args.backend_sample_confirmation_recheck_count))
        ),
        enable_adaptive_candidate_pool=bool(
            getattr(args, "enable_adaptive_candidate_pool", False)
        ),
        adaptive_candidate_pool_min_size=max(
            1,
            int(getattr(args, "adaptive_candidate_pool_min_size", None) or 4),
        ),
        adaptive_candidate_pool_full_sweep_interval=max(
            0,
            int(
                12
                if getattr(args, "adaptive_candidate_pool_full_sweep_interval", None) is None
                else args.adaptive_candidate_pool_full_sweep_interval
            ),
        ),
        adaptive_candidate_pool_calibration_cases=max(
            0,
            int(
                8
                if getattr(args, "adaptive_candidate_pool_calibration_cases", None) is None
                else args.adaptive_candidate_pool_calibration_cases
            ),
        ),
        adaptive_candidate_pool_candidate_burst_cases=max(
            0,
            int(
                8
                if getattr(args, "adaptive_candidate_pool_candidate_burst_cases", None) is None
                else args.adaptive_candidate_pool_candidate_burst_cases
            ),
        ),
        adaptive_candidate_pool_candidate_burst_novel_only=bool(
            getattr(
                args,
                "adaptive_candidate_pool_candidate_burst_novel_only",
                False,
            )
        ),
        adaptive_candidate_pool_preserve_seed_stride=bool(
            getattr(args, "adaptive_candidate_pool_preserve_seed_stride", False)
        ),
        adaptive_candidate_pool_compensate_seed_horizon=bool(
            getattr(
                args,
                "adaptive_candidate_pool_compensate_seed_horizon",
                False,
            )
        ),
        enable_preflight_validation=not args.disable_preflight_validation,
        enable_preflight_repair=not args.disable_preflight_repair,
        persist_feedback_corpus=args.persist_feedback_corpus,
        feedback_persist_limit=max(0, int(getattr(args, "feedback_persist_limit", 4096))),
        enable_local_source_scheduler=bool(getattr(args, "enable_local_source_scheduler", False)),
        local_source_exploration_weight=max(
            0.0, float(getattr(args, "local_source_exploration_weight", 0.5))
        ),
        compress_run_log=not args.no_compress_run_log,
        artifact_limit=args.artifact_limit,
        oracle_mode="both" if args.enable_metamorphic_oracle else "differential",
        generator_profile=args.profile,
        generator_profile_pool=parse_guidance_targets(getattr(args, "profile_pool", "")),
        version_pair_pool=parse_guidance_targets(getattr(args, "version_pair_pool", "")),
        generator_profile_learning_weight=max(
            0.0,
            float(getattr(args, "profile_learning_weight", 0.0) or 0.0),
        ),
        semantic_objective_learning_weight=max(
            0.0,
            float(getattr(args, "semantic_objective_learning_weight", 0.0) or 0.0),
        ),
        metamorphic_relation_learning_weight=max(
            0.0,
            float(getattr(args, "metamorphic_relation_learning_weight", 0.0) or 0.0),
        ),
        version_pair_learning_weight=max(
            0.0,
            float(getattr(args, "version_pair_learning_weight", 0.0) or 0.0),
        ),
        backend_pair_learning_weight=max(
            0.0,
            float(getattr(args, "backend_pair_learning_weight", 0.0) or 0.0),
        ),
        backend_pair_priority_limit=max(
            1,
            int(getattr(args, "backend_pair_priority_limit", 3) or 3),
        ),
        enable_generator_profile_learning="profile_learning" not in disabled_adaptive_components,
        enable_semantic_objective_learning="semantic_objective_learning" not in disabled_adaptive_components,
        enable_metamorphic_relation_learning="metamorphic_relation_learning" not in disabled_adaptive_components,
        enable_backend_pair_learning="backend_pair_learning" not in disabled_adaptive_components,
        enable_profile_capability_filter="profile_capability_filter" not in disabled_adaptive_components,
        enable_mutation_operator_learning="mutation_operator_learning" not in disabled_adaptive_components,
        enable_operator_swarm="operator_swarm" not in disabled_adaptive_components,
        enable_ir_rewrite_mutations="ir_rewrite_mutations" not in disabled_adaptive_components,
        enable_divergence_conditioned_mutations="divergence_conditioned" not in disabled_adaptive_components,
        enable_shrink_mutations="shrink_mutations" not in disabled_adaptive_components,
        enable_value_catalog="value_catalog" not in disabled_adaptive_components,
        enable_quality_archive="quality_archive" not in disabled_adaptive_components,
        enable_hierarchical_archive="hierarchical_archive" not in disabled_adaptive_components,
        enable_bd_axis_bandit="bd_axis_bandit" not in disabled_adaptive_components,
        enable_bayesian_exploration="bayesian_exploration" not in disabled_adaptive_components,
        enable_seed_quota="seed_quota" not in disabled_adaptive_components,
        enable_seed_energy_batch="seed_energy_batch" not in disabled_adaptive_components,
        enable_seed_energy_tier_bandit="seed_energy_tier" not in disabled_adaptive_components,
        enable_per_operator_energy="per_operator_energy" not in disabled_adaptive_components,
        enable_lineage_rarity="lineage_rarity" not in disabled_adaptive_components,
        enable_minhash_dedup="minhash_dedup" not in disabled_adaptive_components,
        enable_disagreement_bd_axis="disagreement_bd_axis" not in disabled_adaptive_components,
        enable_lhs_seeding="lhs_seeding" not in disabled_adaptive_components,
        enable_champion_corpus="champion_corpus" not in disabled_adaptive_components,
        enable_champion_graft_donor_bandit="champion_graft_donor" not in disabled_adaptive_components,
        guidance_strategy=getattr(args, "strategy", "random"),
        guidance_candidate_pool=max(1, int(getattr(args, "candidate_pool", 1))),
        guidance_targets=parse_guidance_targets(getattr(args, "targets", "")),
        exploration_objective_rules=parse_exploration_objective_rules(
            getattr(args, "exploration_objective_rules", "")
        ),
        enable_family_saturation=not bool(getattr(args, "disable_family_saturation", False)),
        family_saturation_threshold=max(1, int(getattr(args, "family_saturation_threshold", 8))),
        family_saturation_penalty=max(0.0, float(getattr(args, "family_saturation_penalty", 1.25))),
        saturated_family_reward=max(0.0, float(getattr(args, "saturated_family_reward", 0.02))),
        known_saturated_bug_families=parse_guidance_targets(
            getattr(args, "known_saturated_bug_families", "")
        ),
        replay_bug_source_issues=(
            parse_guidance_targets(getattr(args, "replay_bug_source_issues", ""))
            or list(DEFAULT_REPLAY_BUG_SOURCE_ISSUES)
        ),
        issue_replay_saturation_threshold=max(1, int(getattr(args, "issue_replay_saturation_threshold", 1))),
        issue_replay_saturation_penalty=max(0.0, float(getattr(args, "issue_replay_saturation_penalty", 1.0))),
        issue_replay_global_saturation_threshold=max(
            1,
            int(getattr(args, "issue_replay_global_saturation_threshold", 4)),
        ),
        issue_replay_global_saturation_penalty=max(
            0.0,
            float(getattr(args, "issue_replay_global_saturation_penalty", 1.5)),
        ),
        issue_inspired_source_saturation_threshold=max(
            1,
            int(getattr(args, "issue_inspired_source_saturation_threshold", 3)),
        ),
        issue_inspired_source_saturation_penalty=max(
            0.0,
            float(getattr(args, "issue_inspired_source_saturation_penalty", 1.25)),
        ),
        candidate_recheck_count=max(0, int(getattr(args, "candidate_recheck_count", 0))),
        metamorphic_variant_limit=max(0, int(getattr(args, "metamorphic_variant_limit", 4))),
        metamorphic_relation_order=parse_guidance_targets(getattr(args, "metamorphic_relation_order", "")),
        target_version=str(getattr(args, "target_version", "") or ""),
        fixed_version=str(getattr(args, "fixed_version", "") or ""),
        log_level=getattr(args, "log_level", "compact"),
        strategy_snapshot_path=str(getattr(args, "strategy_snapshot", "") or ""),
        strategy_learning_path=str(getattr(args, "strategy_learning", "") or ""),
        freeze_strategy_snapshot=bool(getattr(args, "freeze_strategy_snapshot", False)),
    )


def apply_adaptive_component_config(config: ExperimentConfig, disabled_components: set[str]) -> None:
    config.enable_generator_profile_learning = "profile_learning" not in disabled_components
    config.enable_semantic_objective_learning = "semantic_objective_learning" not in disabled_components
    config.enable_metamorphic_relation_learning = "metamorphic_relation_learning" not in disabled_components
    config.enable_backend_pair_learning = "backend_pair_learning" not in disabled_components
    config.enable_profile_capability_filter = "profile_capability_filter" not in disabled_components
    config.enable_mutation_operator_learning = "mutation_operator_learning" not in disabled_components
    config.enable_operator_swarm = "operator_swarm" not in disabled_components
    config.enable_ir_rewrite_mutations = "ir_rewrite_mutations" not in disabled_components
    config.enable_divergence_conditioned_mutations = "divergence_conditioned" not in disabled_components
    config.enable_shrink_mutations = "shrink_mutations" not in disabled_components
    config.enable_value_catalog = "value_catalog" not in disabled_components
    config.enable_quality_archive = "quality_archive" not in disabled_components
    config.enable_hierarchical_archive = "hierarchical_archive" not in disabled_components
    config.enable_bd_axis_bandit = "bd_axis_bandit" not in disabled_components
    config.enable_bayesian_exploration = "bayesian_exploration" not in disabled_components
    config.enable_seed_quota = "seed_quota" not in disabled_components
    config.enable_seed_energy_batch = "seed_energy_batch" not in disabled_components
    config.enable_seed_energy_tier_bandit = "seed_energy_tier" not in disabled_components
    config.enable_per_operator_energy = "per_operator_energy" not in disabled_components
    config.enable_lineage_rarity = "lineage_rarity" not in disabled_components
    config.enable_minhash_dedup = "minhash_dedup" not in disabled_components
    config.enable_disagreement_bd_axis = "disagreement_bd_axis" not in disabled_components
    config.enable_lhs_seeding = "lhs_seeding" not in disabled_components
    config.enable_champion_corpus = "champion_corpus" not in disabled_components
    config.enable_champion_graft_donor_bandit = "champion_graft_donor" not in disabled_components
    if "profile_learning" in disabled_components:
        config.generator_profile_learning_weight = 0.0
    if "semantic_objective_learning" in disabled_components:
        config.semantic_objective_learning_weight = 0.0
    if "metamorphic_relation_learning" in disabled_components:
        config.metamorphic_relation_learning_weight = 0.0
    if "version_pair_learning" in disabled_components:
        config.version_pair_learning_weight = 0.0
    if "backend_pair_learning" in disabled_components:
        config.backend_pair_learning_weight = 0.0
    if "local_source_scheduler" in disabled_components:
        config.enable_local_source_scheduler = False
