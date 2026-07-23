from __future__ import annotations

from typing import Any

from datadiff.champion_corpus import ChampionRegistry
from datadiff.config import ExperimentConfig
from datadiff.feedback import FeedbackState
from datadiff.guidance import GuidanceState
from datadiff.run_config import _effective_guidance_targets
from datadiff.source_scheduler import LocalSourceScheduler


def _restore_closed_loop_state(
    closed_loop_state: dict[str, Any] | None,
    *,
    config: ExperimentConfig,
    backends: list[str],
    guidance_enabled: bool,
    feedback_enabled: bool,
    champion_registry: ChampionRegistry | None = None,
    champion_version_id: str = "",
    feedback_state_cls: type = FeedbackState,
    guidance_state_cls: type = GuidanceState,
    source_scheduler_cls: type = LocalSourceScheduler,
) -> tuple[set[str], set[str], Any | None, Any | None]:
    state = closed_loop_state if isinstance(closed_loop_state, dict) else {}
    seen = {str(item) for item in state.get("seen_signatures", []) or []}
    signal_seen = {str(item) for item in state.get("signal_seen_signatures", []) or []}
    if not signal_seen:
        signal_seen = set(seen)
    feedback = None
    guidance = None
    if feedback_enabled:
        corpus_mode = config.method_policy.generation.corpus_mode
        scheduler_state = _restore_source_scheduler_state(
            state,
            config=config,
            source_scheduler_cls=source_scheduler_cls,
        )
        raw_feedback_state = state.get("feedback")
        if isinstance(raw_feedback_state, dict) and callable(getattr(feedback_state_cls, "from_state_dict", None)):
            feedback = feedback_state_cls.from_state_dict(
                raw_feedback_state,
                persist_to_disk=config.persist_feedback_corpus,
                max_persisted=config.feedback_persist_limit,
                max_cases_per_profile=config.feedback_max_cases_per_profile,
                source_scheduler=scheduler_state,
                corpus_mode=corpus_mode,
                enable_mutation_operator_learning=config.enable_mutation_operator_learning,
                enable_operator_swarm=config.enable_operator_swarm,
                enable_ir_rewrite_mutations=config.enable_ir_rewrite_mutations,
                enable_divergence_conditioned_mutations=config.enable_divergence_conditioned_mutations,
                enable_shrink_mutations=config.enable_shrink_mutations,
                enable_value_catalog=config.enable_value_catalog,
                enable_quality_archive=config.enable_quality_archive,
                enable_hierarchical_archive=config.enable_hierarchical_archive,
                enable_bd_axis_bandit=config.enable_bd_axis_bandit,
                enable_bayesian_exploration=config.enable_bayesian_exploration,
                enable_seed_quota=config.enable_seed_quota,
                enable_seed_energy_batch=config.enable_seed_energy_batch,
                enable_seed_energy_tier_bandit=config.enable_seed_energy_tier_bandit,
                enable_per_operator_energy=config.enable_per_operator_energy,
                enable_lineage_rarity=config.enable_lineage_rarity,
                enable_minhash_dedup=config.enable_minhash_dedup,
                enable_disagreement_bd_axis=config.enable_disagreement_bd_axis,
                enable_champion_corpus=config.enable_champion_corpus,
                enable_champion_graft_donor_bandit=config.enable_champion_graft_donor_bandit,
            )
            feedback.champion_registry = champion_registry if config.enable_champion_corpus else None
            feedback.champion_version_id = champion_version_id
        else:
            feedback = feedback_state_cls(
                persist_to_disk=config.persist_feedback_corpus,
                max_persisted=config.feedback_persist_limit,
                max_cases_per_profile=config.feedback_max_cases_per_profile,
                source_scheduler=scheduler_state,
                corpus_mode=corpus_mode,
                enable_mutation_operator_learning=config.enable_mutation_operator_learning,
                enable_operator_swarm=config.enable_operator_swarm,
                enable_ir_rewrite_mutations=config.enable_ir_rewrite_mutations,
                enable_divergence_conditioned_mutations=config.enable_divergence_conditioned_mutations,
                enable_shrink_mutations=config.enable_shrink_mutations,
                enable_value_catalog=config.enable_value_catalog,
                enable_quality_archive=config.enable_quality_archive,
                enable_hierarchical_archive=config.enable_hierarchical_archive,
                enable_bd_axis_bandit=config.enable_bd_axis_bandit,
                enable_bayesian_exploration=config.enable_bayesian_exploration,
                enable_seed_quota=config.enable_seed_quota,
                enable_seed_energy_batch=config.enable_seed_energy_batch,
                enable_seed_energy_tier_bandit=config.enable_seed_energy_tier_bandit,
                enable_per_operator_energy=config.enable_per_operator_energy,
                enable_lineage_rarity=config.enable_lineage_rarity,
                enable_minhash_dedup=config.enable_minhash_dedup,
                enable_disagreement_bd_axis=config.enable_disagreement_bd_axis,
                enable_champion_corpus=config.enable_champion_corpus,
                enable_champion_graft_donor_bandit=config.enable_champion_graft_donor_bandit,
                champion_registry=champion_registry if config.enable_champion_corpus else None,
                champion_version_id=champion_version_id,
            )
    if guidance_enabled:
        guidance = _restore_guidance_state(
            state.get("guidance"),
            config=config,
            backends=backends,
            guidance_state_cls=guidance_state_cls,
        )
    return seen, signal_seen, feedback, guidance


def _restore_source_scheduler_state(
    state: dict[str, Any],
    *,
    config: ExperimentConfig,
    source_scheduler_cls: type = LocalSourceScheduler,
) -> Any | None:
    if not config.enable_local_source_scheduler:
        return None
    raw_scheduler_state = state.get("source_scheduler")
    if not isinstance(raw_scheduler_state, dict):
        raw_feedback_state = state.get("feedback")
        if isinstance(raw_feedback_state, dict):
            raw_scheduler_state = raw_feedback_state.get("source_scheduler")
    if isinstance(raw_scheduler_state, dict) and callable(getattr(source_scheduler_cls, "from_state_dict", None)):
        return source_scheduler_cls.from_state_dict(
            raw_scheduler_state,
            exploration_weight=config.local_source_exploration_weight,
            enable_family_saturation=config.enable_family_saturation,
            family_saturation_threshold=config.family_saturation_threshold,
            saturated_family_reward=config.saturated_family_reward,
            known_saturated_bug_families=config.known_saturated_bug_families,
        )
    return source_scheduler_cls(
        exploration_weight=config.local_source_exploration_weight,
        enable_family_saturation=config.enable_family_saturation,
        family_saturation_threshold=config.family_saturation_threshold,
        saturated_family_reward=config.saturated_family_reward,
        known_saturated_bug_families=config.known_saturated_bug_families,
    )


def _restore_guidance_state(
    raw_guidance_state: Any,
    *,
    config: ExperimentConfig,
    backends: list[str],
    guidance_state_cls: type = GuidanceState,
) -> Any:
    kwargs = {
        "targets": _effective_guidance_targets(config),
        "discovery_biases": list(config.discovery_biases),
        "exploration_objective_rules": list(config.exploration_objective_rules),
        "enable_family_saturation": config.enable_family_saturation,
        "family_saturation_threshold": config.family_saturation_threshold,
        "family_saturation_penalty": config.family_saturation_penalty,
        "saturated_family_reward": config.saturated_family_reward,
        "known_saturated_bug_families": config.known_saturated_bug_families,
        "issue_replay_saturation_threshold": config.issue_replay_saturation_threshold,
        "issue_replay_saturation_penalty": config.issue_replay_saturation_penalty,
        "issue_replay_global_saturation_threshold": config.issue_replay_global_saturation_threshold,
        "issue_replay_global_saturation_penalty": config.issue_replay_global_saturation_penalty,
        "issue_inspired_source_saturation_threshold": config.issue_inspired_source_saturation_threshold,
        "issue_inspired_source_saturation_penalty": config.issue_inspired_source_saturation_penalty,
        "active_backends": list(backends),
    }
    if isinstance(raw_guidance_state, dict) and callable(getattr(guidance_state_cls, "from_state_dict", None)):
        return guidance_state_cls.from_state_dict(raw_guidance_state, **kwargs)
    return guidance_state_cls(**kwargs)


def _build_closed_loop_state(
    *,
    seen: set[str],
    signal_seen: set[str],
    feedback: Any | None,
    guidance: Any | None,
) -> dict[str, Any]:
    source_scheduler = getattr(feedback, "source_scheduler", None) if feedback is not None else None
    return {
        "seen_signatures": sorted(seen),
        "signal_seen_signatures": sorted(signal_seen),
        "feedback": feedback.to_state_dict() if feedback is not None else None,
        "source_scheduler": (
            source_scheduler.to_state_dict()
            if source_scheduler is not None and callable(getattr(source_scheduler, "to_state_dict", None))
            else None
        ),
        "guidance": guidance.to_state_dict() if guidance is not None else None,
    }


def _inject_champion_corpus(
    feedback: Any | None,
    *,
    version_id: str,
    limit: int,
) -> int:
    registry = getattr(feedback, "champion_registry", None)
    if feedback is None or registry is None:
        return 0
    injected = 0
    inject = getattr(feedback, "inject_champion_seed", None)
    if not callable(inject):
        return 0
    for champion in registry.champions_for_version(version_id, limit=limit):
        if inject(champion):
            injected += 1
    return injected
