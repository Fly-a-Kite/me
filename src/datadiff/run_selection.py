from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from datadiff.bandit_selection import (
    _case_learning_context_features,
    _config_for_version_pair,
    _config_payload_for_version_pair,
    _metamorphic_relation_order_from_selection,
    _select_adaptive_action,
    _semantic_objective_pool,
    _unique_nonempty_strings,
    _version_pair_id,
)
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.metamorphic import all_metamorphic_variants
from datadiff.operation_combo import describe_operation_combo
from datadiff.run_metadata import _candidate_quality_context, _selected_candidate_metadata


@dataclass(frozen=True, slots=True)
class SelectedIterationCase:
    case: Case
    selected_meta: dict[str, Any]
    guidance_row: dict[str, Any]
    preflight_row: dict[str, Any]
    case_seed: int
    effective_config: ExperimentConfig
    effective_config_payload: dict[str, Any]
    version_pair_id: str
    scheduler_elapsed_ms: float


def select_iteration_case(
    *,
    candidates: list[Case],
    candidate_meta: dict[int, dict[str, Any]],
    config: ExperimentConfig,
    config_payload: dict[str, Any],
    feedback: Any,
    guidance: Any,
    include_online_weight_snapshot: bool,
    version_pair_pool: tuple[str, ...],
    version_pair_pool_metadata: dict[str, Any],
    version_pair_context_features: tuple[str, ...],
    target_capabilities: list[str] | tuple[str, ...],
    backends: list[str],
) -> SelectedIterationCase:
    started = time.perf_counter()
    case, guidance_row, selected_operation_combo = _select_guided_case(
        candidates,
        guidance=guidance,
        guidance_strategy=config.guidance_strategy,
        include_online_weight_snapshot=include_online_weight_snapshot,
    )
    guidance_elapsed_ms = (time.perf_counter() - started) * 1000

    selected_meta = candidate_meta.get(id(case))
    if selected_meta is None:
        selected_meta = _selected_candidate_metadata(
            case,
            config,
            bool(guidance is not None and config.enable_family_saturation),
        )
    if not selected_meta.get("quality_archive_context"):
        selected_meta["quality_archive_context"] = _candidate_quality_context(
            feedback,
            case,
            selected_meta,
            target_capabilities=target_capabilities,
        )
        if selected_meta["quality_archive_context"]:
            case.metadata["quality_archive_context"] = selected_meta["quality_archive_context"]
    selected_meta["operation_combo"] = selected_operation_combo or describe_operation_combo(case.program.operations)

    selected_version_pair, version_pair_selection = _select_adaptive_action(
        feedback,
        scope="version_pair",
        action_pool=version_pair_pool,
        context_features=version_pair_context_features,
        version_id=_version_pair_id(config),
        learning_weight=config.version_pair_learning_weight,
        enabled=bool(version_pair_pool),
    )
    effective_config = _config_for_version_pair(config, selected_version_pair)
    effective_config_payload = _config_payload_for_version_pair(config_payload, selected_version_pair)
    version_pair_id = _version_pair_id(effective_config)
    case_learning_context = _case_learning_context_features(
        case,
        effective_config,
        backends=backends,
        target_capabilities=target_capabilities,
        operation_combo=selected_meta["operation_combo"],
        guidance_row=guidance_row,
    )
    semantic_objective_pool = _semantic_objective_pool(case_learning_context, config, guidance_row)
    selected_semantic_objective, semantic_objective_selection = _select_adaptive_action(
        feedback,
        scope="semantic_objective",
        action_pool=semantic_objective_pool,
        context_features=case_learning_context,
        version_id=version_pair_id,
        learning_weight=config.semantic_objective_learning_weight,
        enabled=config.enable_semantic_objective_learning,
    )
    all_mr_variants = all_metamorphic_variants(case) if config.enable_metamorphic_oracle else []
    metamorphic_relation_pool = tuple(_unique_nonempty_strings([variant.relation for variant in all_mr_variants]))
    selected_metamorphic_relation, metamorphic_relation_selection = _select_adaptive_action(
        feedback,
        scope="metamorphic_relation",
        action_pool=metamorphic_relation_pool,
        context_features=case_learning_context,
        version_id=version_pair_id,
        learning_weight=config.metamorphic_relation_learning_weight,
        enabled=config.enable_metamorphic_relation_learning and config.enable_metamorphic_oracle,
    )
    selected_meta["semantic_objective_selection"] = semantic_objective_selection
    selected_meta["selected_semantic_objective"] = selected_semantic_objective
    selected_meta["metamorphic_relation_selection"] = metamorphic_relation_selection
    selected_meta["selected_metamorphic_relation"] = selected_metamorphic_relation
    selected_meta["version_pair_selection"] = version_pair_selection
    selected_meta["selected_version_pair"] = selected_version_pair
    selected_meta["version_pair_pool_metadata"] = dict(version_pair_pool_metadata)
    selected_meta["case_learning_context"] = list(case_learning_context)
    selected_meta["metamorphic_relation_order"] = _metamorphic_relation_order_from_selection(
        selected_metamorphic_relation,
        configured_order=config.metamorphic_relation_order,
        relation_pool=metamorphic_relation_pool,
        rotation_seed=case.seed,
        selection_strategy=str(metamorphic_relation_selection.get("strategy", "") or ""),
    )
    activate_selected = getattr(feedback, "activate_selected_candidate", None)
    if callable(activate_selected):
        activate_selected(case, selected_meta)
    return SelectedIterationCase(
        case=case,
        selected_meta=selected_meta,
        guidance_row=guidance_row,
        preflight_row=selected_meta["preflight"],
        case_seed=case.seed,
        effective_config=effective_config,
        effective_config_payload=effective_config_payload,
        version_pair_id=version_pair_id,
        scheduler_elapsed_ms=guidance_elapsed_ms,
    )


def _select_guided_case(
    candidates: list[Case],
    *,
    guidance: Any,
    guidance_strategy: str,
    include_online_weight_snapshot: bool,
) -> tuple[Case, dict[str, Any], dict[str, Any] | None]:
    if not candidates:
        raise ValueError("select_iteration_case requires at least one candidate")
    if guidance is None:
        return (
            candidates[0],
            {
                "strategy": guidance_strategy,
                "score": 0.0,
                "features": [],
                "matched_targets": [],
                "candidate_count": 1,
            },
            None,
        )
    guidance_selector = getattr(guidance, "select_case", None) or getattr(guidance, "choose_case")
    decision = guidance_selector(
        candidates,
        include_online_weight_snapshot=include_online_weight_snapshot,
    )
    guidance_row = decision.to_dict()
    guidance_row["strategy"] = guidance_strategy
    selected_operation_combo = (
        decision.analysis.operation_combo
        if getattr(decision, "analysis", None) is not None
        else None
    )
    return decision.case, guidance_row, selected_operation_combo
