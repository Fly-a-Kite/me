from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from datadiff.bandit_selection import _select_generator_profile
from datadiff.case_policy import known_replay_source_filter_reason
from datadiff.case_policy import case_discovery_origin
from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.boundary_values import apply_targeted_boundary_profile
from datadiff.dsl import Case
from datadiff.goal_first import (
    generate_goal_first_case,
    refresh_goal_semantic_activation,
    semantic_witness_epoch_key,
)
from datadiff.metamorphic import all_metamorphic_variants
from datadiff.preflight import preflight_case
from datadiff.run_metadata import _candidate_quality_context, _generated_candidate_metadata
from datadiff.family_witness_registry import (
    generate_registered_family_witness_case,
)

REPLAY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE = 20
SATURATED_FAMILY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE = 20

GenerateCaseFn = Callable[..., Case]
SchemaSpecFn = Callable[[int], Any | None]
ReplayFilterFn = Callable[[Case, ExperimentConfig], str]


def _known_replay_source_filter_reason(case_item: Case, config: ExperimentConfig) -> str:
    return known_replay_source_filter_reason(
        case_item,
        enable_replay_bug=config.enable_replay_bug,
        replay_bug_source_issues=config.replay_bug_source_issues,
    )


_replay_bug_filter_reason = _known_replay_source_filter_reason


@dataclass(slots=True)
class CandidateBatch:
    candidates: list[Case]
    candidate_meta: dict[int, dict[str, Any]]
    next_seed: int
    replay_filtered_candidates: int = 0
    replay_fallback_candidates: int = 0
    saturated_family_filtered_candidates: int = 0
    saturated_family_fallback_candidates: int = 0

    def counter_deltas(self) -> dict[str, int]:
        return {
            "replay_filtered_candidates": self.replay_filtered_candidates,
            "replay_fallback_candidates": self.replay_fallback_candidates,
            "saturated_family_filtered_candidates": self.saturated_family_filtered_candidates,
            "saturated_family_fallback_candidates": self.saturated_family_fallback_candidates,
        }


@dataclass(slots=True)
class _CandidateBuildState:
    seed_cursor: int
    generation_epoch_seed: int | None
    replay_filtered_candidates: int = 0
    replay_fallback_candidates: int = 0
    saturated_family_filtered_candidates: int = 0
    saturated_family_fallback_candidates: int = 0
    candidates: list[Case] = field(default_factory=list)
    candidate_meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    reserved_generator_profiles: set[str] = field(default_factory=set)


def generate_candidate_batch(
    *,
    seed_start: int,
    candidate_pool: int,
    config: ExperimentConfig,
    feedback: Any,
    guidance: Any,
    generator_profile_pool: tuple[str, ...],
    generator_profile_pool_metadata: dict[str, Any],
    generator_profile_context_features: tuple[str, ...],
    target_capabilities: list[str] | tuple[str, ...],
    schema_spec_for_seed: SchemaSpecFn,
    generate_case_fn: GenerateCaseFn = generate_case,
    replay_filter_fn: ReplayFilterFn = _known_replay_source_filter_reason,
    force_fresh_source: bool = False,
    generation_epoch_seed: int | None = None,
) -> CandidateBatch:
    resolved_generation_epoch_seed = generation_epoch_seed
    if (
        resolved_generation_epoch_seed is None
        and config.method_policy.generation.mode
        in {"goal_first_witness_v2", "goal_first_witness_v3"}
    ):
        resolved_generation_epoch_seed = semantic_witness_epoch_key(seed_start)
    state = _CandidateBuildState(
        seed_cursor=int(seed_start),
        generation_epoch_seed=(
            None
            if resolved_generation_epoch_seed is None
            else int(resolved_generation_epoch_seed)
        ),
    )
    target_pool = max(1, int(candidate_pool))
    while len(state.candidates) < target_pool:
        _append_candidate(
            state,
            config=config,
            feedback=feedback,
            guidance=guidance,
            generator_profile_pool=generator_profile_pool,
            generator_profile_pool_metadata=generator_profile_pool_metadata,
            generator_profile_context_features=generator_profile_context_features,
            target_capabilities=target_capabilities,
            candidate_slots_remaining=target_pool - len(state.candidates),
            schema_spec_for_seed=schema_spec_for_seed,
            generate_case_fn=generate_case_fn,
            replay_filter_fn=replay_filter_fn,
            force_fresh_source=force_fresh_source,
        )
    return CandidateBatch(
        candidates=state.candidates,
        candidate_meta=state.candidate_meta,
        next_seed=state.seed_cursor,
        replay_filtered_candidates=state.replay_filtered_candidates,
        replay_fallback_candidates=state.replay_fallback_candidates,
        saturated_family_filtered_candidates=state.saturated_family_filtered_candidates,
        saturated_family_fallback_candidates=state.saturated_family_fallback_candidates,
    )


def _append_candidate(
    state: _CandidateBuildState,
    *,
    config: ExperimentConfig,
    feedback: Any,
    guidance: Any,
    generator_profile_pool: tuple[str, ...],
    generator_profile_pool_metadata: dict[str, Any],
    generator_profile_context_features: tuple[str, ...],
    target_capabilities: list[str] | tuple[str, ...],
    candidate_slots_remaining: int,
    schema_spec_for_seed: SchemaSpecFn,
    generate_case_fn: GenerateCaseFn,
    replay_filter_fn: ReplayFilterFn,
    force_fresh_source: bool,
) -> None:
    skipped_replay_candidates = 0
    last_replay_skip_reason = ""
    replay_fallback_used = False
    skipped_saturated_family_candidates = 0
    last_saturated_family_skip_reason = ""
    saturated_family_fallback_used = False
    rejected_profiles: set[str] = set()
    while True:
        case_seed = state.seed_cursor
        state.seed_cursor += 1
        available_profiles = tuple(
            profile
            for profile in generator_profile_pool
            if profile not in state.reserved_generator_profiles
            and profile not in rejected_profiles
        )
        if not available_profiles:
            available_profiles = tuple(
                profile
                for profile in generator_profile_pool
                if profile not in rejected_profiles
            ) or generator_profile_pool
        rotation = case_seed % len(available_profiles)
        available_profiles = (
            *available_profiles[rotation:],
            *available_profiles[:rotation],
        )
        selected_profile, profile_selection = _select_generator_profile(
            feedback,
            available_profiles,
            context_features=generator_profile_context_features,
            learning_weight=(
                config.generator_profile_learning_weight
                if config.enable_generator_profile_learning
                else 0.0
            ),
            pool_metadata=generator_profile_pool_metadata,
        )
        pending = None if force_fresh_source else _pop_pending_feedback_candidate(feedback)
        if pending is None:
            generated = _generate_case_with_optional_schema(
                case_seed,
                type_aware=config.enable_type_aware_generation,
                profile=selected_profile,
                schema_spec=schema_spec_for_seed(case_seed),
                generation_mode=config.method_policy.generation.mode,
                boundary_mode=config.method_policy.generation.boundary_mode,
                witness_epoch_seed=state.generation_epoch_seed,
                generate_case_fn=generate_case_fn,
            )
            generated_replay_skip_reason = replay_filter_fn(generated, config)
            if generated_replay_skip_reason:
                rejected_profiles.add(selected_profile)
                last_replay_skip_reason = generated_replay_skip_reason
                state.replay_filtered_candidates += 1
                skipped_replay_candidates += 1
                if skipped_replay_candidates <= REPLAY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE:
                    continue
                replay_fallback_used = True
                state.replay_fallback_candidates += 1
                (
                    case_seed,
                    candidate,
                    source,
                    metadata,
                    profile_selection,
                    preflight,
                    last_saturated_family_skip_reason,
                ) = _fallback_candidate(
                    state,
                    config=config,
                    guidance=None,
                    generator_profile_pool=generator_profile_pool,
                    generator_profile_pool_metadata=generator_profile_pool_metadata,
                    schema_spec_for_seed=schema_spec_for_seed,
                    generate_case_fn=generate_case_fn,
                    replay_filter_fn=replay_filter_fn,
                    strategy="fresh_fallback",
                    source="generated_fresh_fallback",
                    replay_error_prefix="fresh fallback generated replay candidate",
                    previous_saturation_reason=last_saturated_family_skip_reason,
                )
                break
            selected, source, metadata = _select_feedback_candidate(
                None if force_fresh_source else feedback,
                case_seed,
                generated,
                max_batch=candidate_slots_remaining,
            )
        else:
            selected, source, metadata = pending
        selected, source, metadata = _resolve_explicit_candidate_source(
            selected,
            source=source,
            metadata=metadata,
            seed=case_seed,
            config=config,
            force_fresh_source=force_fresh_source,
        )
        preflight = preflight_case(
            selected,
            enable_validation=config.enable_preflight_validation,
            enable_repair=config.enable_preflight_repair,
        )
        candidate = preflight.case
        replay_skip_reason = replay_filter_fn(candidate, config)
        if not replay_skip_reason:
            saturated_roots = _candidate_saturated_roots(guidance, candidate)
            if not saturated_roots:
                break
            last_saturated_family_skip_reason = ",".join(saturated_roots)
            rejected_profiles.add(selected_profile)
            state.saturated_family_filtered_candidates += 1
            skipped_saturated_family_candidates += 1
            if skipped_saturated_family_candidates <= SATURATED_FAMILY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE:
                continue
            saturated_family_fallback_used = True
            state.saturated_family_fallback_candidates += 1
            (
                case_seed,
                candidate,
                source,
                metadata,
                profile_selection,
                preflight,
                last_saturated_family_skip_reason,
            ) = _fallback_candidate(
                state,
                config=config,
                guidance=guidance,
                generator_profile_pool=generator_profile_pool,
                generator_profile_pool_metadata=generator_profile_pool_metadata,
                schema_spec_for_seed=schema_spec_for_seed,
                generate_case_fn=generate_case_fn,
                replay_filter_fn=replay_filter_fn,
                strategy="saturation_fallback",
                source="generated_saturation_fallback",
                replay_error_prefix="saturation fallback generated replay candidate",
                previous_saturation_reason=last_saturated_family_skip_reason,
            )
            break
        last_replay_skip_reason = replay_skip_reason
        rejected_profiles.add(selected_profile)
        state.replay_filtered_candidates += 1
        skipped_replay_candidates += 1
        if skipped_replay_candidates <= REPLAY_FILTER_EXTRA_ATTEMPTS_PER_CANDIDATE:
            continue
        replay_fallback_used = True
        state.replay_fallback_candidates += 1
        (
            case_seed,
            candidate,
            source,
            metadata,
            profile_selection,
            preflight,
            last_saturated_family_skip_reason,
        ) = _fallback_candidate(
            state,
            config=config,
            guidance=None,
            generator_profile_pool=generator_profile_pool,
            generator_profile_pool_metadata=generator_profile_pool_metadata,
            schema_spec_for_seed=schema_spec_for_seed,
            generate_case_fn=generate_case_fn,
            replay_filter_fn=replay_filter_fn,
            strategy="fresh_fallback",
            source="generated_fresh_fallback",
            replay_error_prefix="fresh fallback generated replay candidate",
            previous_saturation_reason=last_saturated_family_skip_reason,
        )
        break
    semantic_activation = refresh_goal_semantic_activation(candidate)
    quality_archive_context = _candidate_quality_context(
        feedback,
        candidate,
        metadata,
        target_capabilities=target_capabilities,
    )
    if quality_archive_context:
        candidate.metadata["quality_archive_context"] = quality_archive_context
    state.candidate_meta[id(candidate)] = {
        "source": source,
        "generated_seed": case_seed,
        "seed_lineage": metadata.get("seed_lineage", {}),
        "mutation": metadata.get("mutation", {}),
        "feedback_decision": metadata.get("feedback_decision", {}),
        "quality_archive_context": quality_archive_context,
        "generator_profile_selection": profile_selection,
        "goal_first_generation": dict(
            candidate.metadata.get("goal_first_generation", {})
            if isinstance(candidate.metadata, dict)
            else {}
        ),
        "semantic_activation": dict(semantic_activation),
        "boundary_application": dict(
            candidate.metadata.get("boundary_application", {})
            if isinstance(candidate.metadata, dict)
            else {}
        ),
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
    reserved_profile = str(profile_selection.get("profile", "") or "").strip()
    if reserved_profile:
        state.reserved_generator_profiles.add(reserved_profile)
    state.candidates.append(candidate)


def _fallback_candidate(
    state: _CandidateBuildState,
    *,
    config: ExperimentConfig,
    guidance: Any,
    generator_profile_pool: tuple[str, ...],
    generator_profile_pool_metadata: dict[str, Any],
    schema_spec_for_seed: SchemaSpecFn,
    generate_case_fn: GenerateCaseFn,
    replay_filter_fn: ReplayFilterFn,
    strategy: str,
    source: str,
    replay_error_prefix: str,
    previous_saturation_reason: str,
) -> tuple[int, Case, str, dict[str, Any], dict[str, Any], Any, str]:
    case_seed = state.seed_cursor
    state.seed_cursor += 1
    generated = _generate_case_with_optional_schema(
        case_seed,
        type_aware=config.enable_type_aware_generation,
        profile="common",
        schema_spec=schema_spec_for_seed(case_seed),
        generation_mode=config.method_policy.generation.mode,
        boundary_mode=config.method_policy.generation.boundary_mode,
        witness_epoch_seed=state.generation_epoch_seed,
        generate_case_fn=generate_case_fn,
    )
    profile_selection = _fallback_profile_selection(
        strategy,
        generator_profile_pool=generator_profile_pool,
        generator_profile_pool_metadata=generator_profile_pool_metadata,
    )
    selected = generated
    metadata = _generated_candidate_metadata(generated)
    preflight = preflight_case(
        selected,
        enable_validation=config.enable_preflight_validation,
        enable_repair=config.enable_preflight_repair,
    )
    candidate = preflight.case
    replay_skip_reason = replay_filter_fn(candidate, config)
    if replay_skip_reason:
        state.replay_filtered_candidates += 1
        raise RuntimeError(f"{replay_error_prefix}: {replay_skip_reason}")
    last_saturated_family_skip_reason = previous_saturation_reason
    fallback_roots = _candidate_saturated_roots(guidance, candidate)
    if fallback_roots:
        last_saturated_family_skip_reason = ",".join(fallback_roots)
    return (
        case_seed,
        candidate,
        source,
        metadata,
        profile_selection,
        preflight,
        last_saturated_family_skip_reason,
    )


def _select_feedback_candidate(
    feedback: Any,
    case_seed: int,
    generated: Case,
    *,
    max_batch: int = 1,
) -> tuple[Case, str, dict[str, Any]]:
    if feedback is None:
        return generated, "generated", _generated_candidate_metadata(generated)
    batch_selector = getattr(feedback, "select_case_batch", None)
    if callable(batch_selector):
        selected_batch = batch_selector(
            case_seed,
            generated,
            max_batch=max_batch,
            enqueue_remaining=max(1, int(max_batch or 1)) > 1,
        )
        selected = selected_batch[0] if selected_batch else generated
    else:
        feedback_selector = getattr(feedback, "select_case", None)
        if not callable(feedback_selector):
            feedback_selector = getattr(feedback, "choose_case", None)
        if not callable(feedback_selector):
            raise AttributeError("feedback state must provide select_case/select_case_batch")
        selected = feedback_selector(case_seed, generated)
    source = getattr(feedback, "last_candidate_source", "generated")
    metadata = (
        getattr(feedback, "last_candidate_metadata", None)
        or selected.metadata
        or _generated_candidate_metadata(generated)
    )
    return selected, source, metadata


def _resolve_explicit_candidate_source(
    case: Case,
    *,
    source: str,
    metadata: dict[str, Any],
    seed: int,
    config: ExperimentConfig,
    force_fresh_source: bool,
) -> tuple[Case, str, dict[str, Any]]:
    """Materialize non-feedback campaign sources with explicit provenance."""

    resolved_source = str(source or "generated")
    resolved_metadata = dict(metadata or {})
    if force_fresh_source:
        return case, resolved_source, resolved_metadata
    if (
        config.enable_known_regression_source
        and resolved_source == "generated"
        and case_discovery_origin(case) == "issue_replay"
    ):
        return case, "known_regression_replay", _source_metadata(
            resolved_metadata,
            source="known_regression_replay",
            seed=seed,
            detail="known_issue_replay",
        )
    if (
        config.enable_semantic_metamorphic_source
        and resolved_source == "generated"
    ):
        variants = all_metamorphic_variants(case)
        if variants:
            variant = variants[seed % len(variants)]
            return variant.case, "semantic_metamorphic_mutation", _source_metadata(
                resolved_metadata,
                source="semantic_metamorphic_mutation",
                seed=seed,
                detail=variant.name,
                relation=variant.relation,
                parent=case,
            )
    return case, resolved_source, resolved_metadata


def _source_metadata(
    metadata: dict[str, Any],
    *,
    source: str,
    seed: int,
    detail: str,
    relation: str = "",
    parent: Case | None = None,
) -> dict[str, Any]:
    out = dict(metadata)
    lineage = dict(out.get("seed_lineage", {}) or {})
    if parent is not None:
        lineage.update(
            {
                "root_seed": parent.seed,
                "parent_seed": parent.seed,
                "parent_case_id": parent.case_id,
                "mutation_seed": seed,
                "depth": int(lineage.get("depth", 0) or 0) + 1,
            }
        )
    out["source"] = source
    out["seed_lineage"] = lineage
    out["mutation"] = {
        "operator": source,
        "detail": detail,
        "relation": relation,
        "changed": True,
    }
    return out


def _pop_pending_feedback_candidate(feedback: Any) -> tuple[Case, str, dict[str, Any]] | None:
    if feedback is None:
        return None
    popper = getattr(feedback, "pop_pending_candidate", None)
    if not callable(popper):
        return None
    selected = popper()
    if selected is None:
        return None
    case = selected.case
    metadata = dict(selected.metadata or case.metadata or {})
    return case, str(selected.source or metadata.get("source", "feedback_mutation") or "feedback_mutation"), metadata


def _candidate_saturated_roots(guidance: Any, candidate: Case) -> list[str]:
    if guidance is None:
        return []
    return list(
        guidance.predicted_saturated_family_roots_for_candidate(
            candidate,
            include_known_families=False,
        )
        or []
    )


def _fallback_profile_selection(
    strategy: str,
    *,
    generator_profile_pool: tuple[str, ...],
    generator_profile_pool_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "profile": "common",
        "profile_pool": list(generator_profile_pool),
        "profile_pool_metadata": dict(generator_profile_pool_metadata),
        "learning_weight": 0.0,
        "ranked": [],
    }


def _generate_case_with_optional_schema(
    seed: int,
    *,
    type_aware: bool,
    profile: str,
    schema_spec: Any | None,
    generation_mode: str = "forward_random",
    boundary_mode: str = "disabled",
    witness_epoch_seed: int | None = None,
    generate_case_fn: GenerateCaseFn = generate_case,
) -> Case:
    family_witness_case = generate_registered_family_witness_case(
        generation_mode,
        seed,
        profile=profile,
    )
    if family_witness_case is not None:
        return family_witness_case
    if generation_mode in {
        "goal_first",
        "goal_first_witness",
        "goal_first_witness_v2",
        "goal_first_witness_v3",
    }:
        generated = generate_goal_first_case(
            seed,
            profile=profile,
            schema_spec=schema_spec,
            boundary_mode=boundary_mode,
            require_semantic_activation=generation_mode
            in {
                "goal_first_witness",
                "goal_first_witness_v2",
                "goal_first_witness_v3",
            },
            diversity_preserving_witness=generation_mode
            in {"goal_first_witness_v2", "goal_first_witness_v3"},
            diversity_epoch_seed=(
                witness_epoch_seed
                if generation_mode
                in {"goal_first_witness_v2", "goal_first_witness_v3"}
                else None
            ),
            witness_variant_family=(
                "v3" if generation_mode == "goal_first_witness_v3" else "v2"
            ),
        )
        if generated.case is not None:
            return generated.case
        fallback = _generate_forward_case(
            seed,
            type_aware=type_aware,
            profile=profile,
            schema_spec=schema_spec,
            generate_case_fn=generate_case_fn,
        )
        fallback.metadata = dict(fallback.metadata or {})
        fallback.metadata["goal_first_generation"] = generated.trace
        fallback.metadata["generation_mode"] = "forward_fallback"
        return fallback
    generated = _generate_forward_case(
        seed,
        type_aware=type_aware,
        profile=profile,
        schema_spec=schema_spec,
        generate_case_fn=generate_case_fn,
    )
    if boundary_mode == "fault_model_targeted":
        return apply_targeted_boundary_profile(generated, seed=seed).case
    return generated


def _generate_forward_case(
    seed: int,
    *,
    type_aware: bool,
    profile: str,
    schema_spec: Any | None,
    generate_case_fn: GenerateCaseFn,
) -> Case:
    if schema_spec is None:
        return generate_case_fn(seed, type_aware=type_aware, profile=profile)
    try:
        return generate_case_fn(
            seed,
            type_aware=type_aware,
            profile=profile,
            schema_spec=schema_spec,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return generate_case_fn(seed, type_aware=type_aware, profile=profile)
