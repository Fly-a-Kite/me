from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff.bandit_selection import _backend_pair_context_features, _rank_backend_pairs
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.quality_oracles import evaluate_quality_oracles
from datadiff.run_metadata import (
    _attach_case_fingerprint_to_row_case,
    _attach_disagreement_descriptor_to_row_case,
    _attach_semantic_contract_lattice_to_row_case,
)
from datadiff.run_signatures import (
    coverage_discovery_signature,
    coverage_signal_signature,
    signal_signature,
)
from datadiff.semantic_contracts import semantic_contract_lattice_payload


@dataclass(frozen=True, slots=True)
class IterationRowUpdate:
    behavior_signature: str
    discovery_signature: str
    signal_signature: str
    quality_oracles: list[dict[str, Any]]


def apply_iteration_row_updates(
    *,
    row: dict[str, Any],
    case: Case,
    selected_meta: dict[str, Any],
    preflight_row: dict[str, Any],
    feedback: Any,
    config: ExperimentConfig,
    guidance_row: dict[str, Any],
    guidance_targets: list[str],
    backend_pair_pool: tuple[str, ...],
    version_pair_id: str,
    target_capabilities: list[str] | tuple[str, ...],
    seen: set[str],
    signal_seen: set[str],
    candidate_seed_start: int,
    candidate_pool: int,
    case_index: int,
    elapsed_s: float,
) -> IterationRowUpdate:
    sig = str(row["behavior_signature"])
    backend_sampling = row.get("backend_sampling", {}) or {}
    coverage_sampled_campaign = bool(backend_sampling.get("sampled", False))
    discovery_sig = (
        coverage_discovery_signature(row)
        if coverage_sampled_campaign
        else str(row.get("discovery_signature", sig))
    )
    if coverage_sampled_campaign:
        row["coverage_discovery_signature"] = discovery_sig
        row["discovery_signature"] = discovery_sig
    row["is_new_behavior"] = discovery_sig not in seen
    seen.add(discovery_sig)
    signal_sig = (
        coverage_signal_signature(row)
        if coverage_sampled_campaign
        else signal_signature(row)
    )
    row["signal_signature"] = signal_sig
    row["signal_new_behavior"] = bool(row["is_new_behavior"]) and signal_sig not in signal_seen
    if row["is_new_behavior"]:
        signal_seen.add(signal_sig)

    _attach_selection_metadata(row, selected_meta)
    disagreement_descriptor = row.get("disagreement_descriptor", {})
    if isinstance(disagreement_descriptor, dict):
        _attach_disagreement_descriptor_to_row_case(row, disagreement_descriptor)
    case_fingerprint = row.get("case_fingerprint", {})
    if isinstance(case_fingerprint, dict):
        _attach_case_fingerprint_to_row_case(row, case_fingerprint)
    semantic_contract_lattice = semantic_contract_lattice_payload(case)
    row["semantic_contract_lattice"] = semantic_contract_lattice
    _attach_semantic_contract_lattice_to_row_case(row, semantic_contract_lattice)

    backend_pair_context = _backend_pair_context_features(
        case_learning_context=tuple(row.get("case_learning_context", []) or ()),
        case_fingerprint=case_fingerprint if isinstance(case_fingerprint, dict) else None,
        disagreement_descriptor=disagreement_descriptor if isinstance(disagreement_descriptor, dict) else None,
        selected_version_pair=row["selected_version_pair"],
        target_capabilities=target_capabilities,
    )
    backend_pair_selection = _rank_backend_pairs(
        feedback,
        pair_pool=backend_pair_pool,
        context_features=backend_pair_context,
        version_id=version_pair_id,
        learning_weight=config.backend_pair_learning_weight,
        enabled=config.enable_backend_pair_learning,
        limit=config.backend_pair_priority_limit,
    )
    row["backend_pair_selection"] = backend_pair_selection
    row["backend_pair_priority"] = list(backend_pair_selection.get("priority", []) or [])
    row["backend_pair_context"] = list(backend_pair_context)
    row_case = row.get("case", {}) if isinstance(row.get("case", {}), dict) else {}
    if (
        isinstance(disagreement_descriptor, dict)
        and isinstance(case.metadata, dict)
        and str(row_case.get("case_id", "") or "") == case.case_id
    ):
        case.metadata["disagreement_descriptor"] = disagreement_descriptor
    if (
        isinstance(case_fingerprint, dict)
        and isinstance(case.metadata, dict)
        and str(row_case.get("case_id", "") or "") == case.case_id
    ):
        case.metadata["case_fingerprint"] = case_fingerprint
    if (
        isinstance(case.metadata, dict)
        and str(row_case.get("case_id", "") or "") == case.case_id
    ):
        case.metadata["semantic_contract_lattice"] = semantic_contract_lattice

    row["operation_combo"] = selected_meta["operation_combo"]
    row["preflight"] = preflight_row
    row["replay_filter"] = selected_meta["replay_filter"]
    row["family_saturation_filter"] = selected_meta["family_saturation_filter"]
    row["candidate_seed_start"] = candidate_seed_start
    row["candidate_pool_size"] = candidate_pool
    row["case_index"] = case_index
    row["elapsed_s"] = float(elapsed_s)
    quality_oracles = [
        oracle.to_dict()
        for oracle in evaluate_quality_oracles(
            case,
            row,
            candidate_source=selected_meta["source"],
            preflight=preflight_row,
            guidance_decision=guidance_row,
            guidance_strategy=config.guidance_strategy,
            guidance_targets=guidance_targets,
            known_saturated_bug_families=config.known_saturated_bug_families,
        )
    ]
    row["quality_oracles"] = quality_oracles
    return IterationRowUpdate(
        behavior_signature=sig,
        discovery_signature=discovery_sig,
        signal_signature=signal_sig,
        quality_oracles=quality_oracles,
    )


def _attach_selection_metadata(row: dict[str, Any], selected_meta: dict[str, Any]) -> None:
    row["candidate_source"] = selected_meta["source"]
    row["seed_lineage"] = selected_meta["seed_lineage"]
    row["mutation"] = selected_meta["mutation"]
    row["feedback_decision"] = selected_meta.get("feedback_decision", {})
    row["quality_archive_context"] = selected_meta.get("quality_archive_context", {})
    row["goal_first_generation"] = selected_meta.get("goal_first_generation", {})
    row["semantic_activation"] = selected_meta.get("semantic_activation", {})
    row["boundary_application"] = selected_meta.get("boundary_application", {})
    row["semantic_novelty"] = selected_meta.get("semantic_novelty", {})
    row["generator_profile_selection"] = selected_meta.get("generator_profile_selection", {})
    row["selected_generator_profile"] = str(row["generator_profile_selection"].get("profile", "") or "")
    row["semantic_objective_selection"] = selected_meta.get("semantic_objective_selection", {})
    row["selected_semantic_objective"] = str(selected_meta.get("selected_semantic_objective", "") or "")
    row["metamorphic_relation_selection"] = selected_meta.get("metamorphic_relation_selection", {})
    row["selected_metamorphic_relation"] = str(selected_meta.get("selected_metamorphic_relation", "") or "")
    row["version_pair_selection"] = selected_meta.get("version_pair_selection", {})
    row["selected_version_pair"] = str(selected_meta.get("selected_version_pair", "") or "")
    row["case_learning_context"] = selected_meta.get("case_learning_context", [])
