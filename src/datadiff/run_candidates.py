from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from datadiff.bandit_selection import _select_generator_profile
from datadiff.case_policy import known_replay_source_filter_reason
from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.dsl import Case
from datadiff.preflight import preflight_case
from datadiff.run_metadata import _candidate_quality_context, _generated_candidate_metadata

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
    replay_filtered_candidates: int = 0
    replay_fallback_candidates: int = 0
    saturated_family_filtered_candidates: int = 0
    saturated_family_fallback_candidates: int = 0
    candidates: list[Case] = field(default_factory=list)
    candidate_meta: dict[int, dict[str, Any]] = field(default_factory=dict)


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
) -> CandidateBatch:
    state = _CandidateBuildState(seed_cursor=int(seed_start))
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
) -> None:
    skipped_replay_candidates = 0
    last_replay_skip_reason = ""
    replay_fallback_used = False
    skipped_saturated_family_candidates = 0
    last_saturated_family_skip_reason = ""
    saturated_family_fallback_used = False
    while True:
        case_seed = state.seed_cursor
        state.seed_cursor += 1
        selected_profile, profile_selection = _select_generator_profile(
            feedback,
            generator_profile_pool,
            context_features=generator_profile_context_features,
            learning_weight=(
                config.generator_profile_learning_weight
                if config.enable_generator_profile_learning
                else 0.0
            ),
            pool_metadata=generator_profile_pool_metadata,
        )
        pending = _pop_pending_feedback_candidate(feedback)
        if pending is None:
            generated = _generate_case_with_optional_schema(
                case_seed,
                type_aware=config.enable_type_aware_generation,
                profile=selected_profile,
                schema_spec=schema_spec_for_seed(case_seed),
                generate_case_fn=generate_case_fn,
            )
            generated_replay_skip_reason = replay_filter_fn(generated, config)
            if generated_replay_skip_reason:
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
                feedback,
                case_seed,
                generated,
                max_batch=candidate_slots_remaining,
            )
        else:
            selected, source, metadata = pending
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
        "feedback_selection": metadata.get("feedback_selection", metadata.get("feedback_decision", {})),
        "feedback_decision": metadata.get("feedback_decision", metadata.get("feedback_selection", {})),
        "quality_archive_context": quality_archive_context,
        "generator_profile_selection": profile_selection,
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
        feedback_selector = getattr(feedback, "select_case", None) or getattr(feedback, "choose_case")
        selected = feedback_selector(case_seed, generated)
    source = getattr(feedback, "last_candidate_source", "generated")
    metadata = (
        getattr(feedback, "last_candidate_metadata", None)
        or selected.metadata
        or _generated_candidate_metadata(generated)
    )
    return selected, source, metadata


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
    generate_case_fn: GenerateCaseFn = generate_case,
) -> Case:
    if schema_spec is None:
        return generate_case_fn(seed, type_aware=type_aware, profile=profile)
    try:
        return generate_case_fn(seed, type_aware=type_aware, profile=profile, schema_spec=schema_spec)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return generate_case_fn(seed, type_aware=type_aware, profile=profile)
