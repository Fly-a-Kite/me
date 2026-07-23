from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from datadiff.method_arms import METHOD_VERSION, P5_METHOD_VERSION, MethodArm


def build_research_control_arms(
    live_arms: Mapping[str, MethodArm],
) -> dict[str, MethodArm]:
    """Construct historical and ablation arms only after explicit selection."""

    arms: dict[str, MethodArm] = {
        "legacy_cartesian": MethodArm(
            arm_id="legacy_cartesian",
            method_version=METHOD_VERSION,
            parent_arm_id="",
            unique_factor="historical_normalized_comparison",
            changed_dimensions=(),
            comparison_mode="legacy",
            observation_mode="legacy",
            ir_mode="legacy_dict",
            obligation_mode="legacy_all",
            obligation_priority_mode="complete_builder_order",
            execution_mode="cartesian",
            selector_mode="all",
            plan_guidance="disabled",
            node_budget=None,
            cache_mode="disabled",
            confirmation_mode="none",
        )
    }

    def child(
        arm_id: str,
        parent_arm_id: str,
        unique_factor: str,
        *,
        method_version: str | None = None,
        **changes: Any,
    ) -> None:
        parent = arms.get(parent_arm_id) or live_arms[parent_arm_id]
        arms[arm_id] = replace(
            parent,
            arm_id=arm_id,
            method_version=method_version or parent.method_version,
            parent_arm_id=parent_arm_id,
            unique_factor=unique_factor,
            changed_dimensions=tuple(changes),
            **changes,
        )

    child(
        "contract_cartesian",
        "legacy_cartesian",
        "lossless_contract_comparison",
        comparison_mode="contract",
        observation_mode="lossless",
    )
    child(
        "contract_ccs_cartesian",
        "contract_cartesian",
        "contract_carrying_semantic_relational_ir",
        ir_mode="ccs_ir",
    )
    child(
        "contract_ccs_obligations_cartesian",
        "contract_ccs_cartesian",
        "executable_ccs_test_obligations",
        obligation_mode="ccs_guided",
    )
    child(
        "contract_ccs_risk_obligations_cartesian",
        "contract_ccs_obligations_cartesian",
        "ccs_static_risk_prioritized_obligations",
        obligation_priority_mode="ccs_risk_priority",
    )
    child(
        "contract_lattice_static",
        "contract_ccs_obligations_cartesian",
        "budgeted_execution_lattice",
        execution_mode="lattice",
        selector_mode="static",
        node_budget=3.0,
        cache_mode="lattice",
    )
    child(
        "contract_lattice_plan",
        "contract_lattice_static",
        "semantic_plan_weighted_subgraph_selection",
        selector_mode="plan",
        plan_guidance="semantic",
    )
    child(
        "contract_lattice_shared_cost",
        "contract_lattice_static",
        "profiled_shared_cost_subgraph_selection",
        selector_mode="shared_cost",
    )
    generated_p4_parent = replace(
        arms["contract_lattice_shared_cost"],
        arm_id="contract_lattice_shared_cost_full",
        parent_arm_id="contract_lattice_shared_cost",
        unique_factor="candidate_triggered_full_backend_confirmation",
        changed_dimensions=("confirmation_mode",),
        confirmation_mode="full",
    )
    _require_same_live_arm(generated_p4_parent, live_arms)
    child(
        "contract_lattice_full",
        "contract_lattice_plan",
        "candidate_triggered_full_confirmation",
        confirmation_mode="full",
    )
    child(
        "p4_lattice_static_cache_off",
        "contract_lattice_static",
        "shared_cost_cache_ablation",
        cache_mode="disabled",
    )
    child(
        "p4_lattice_random_cache_off",
        "p4_lattice_static_cache_off",
        "random_selector_control",
        selector_mode="random",
    )
    child(
        "p4_lattice_shared_cost_cache_off",
        "p4_lattice_static_cache_off",
        "shared_cost_selector_without_cache",
        selector_mode="shared_cost",
    )
    child(
        "p4_lattice_plan_unguided_cache_off",
        "p4_lattice_static_cache_off",
        "plan_selector_without_guidance",
        selector_mode="plan",
    )
    child(
        "p4_lattice_plan_cache_off",
        "contract_lattice_plan",
        "semantic_plan_selector_without_cache",
        cache_mode="disabled",
    )

    p5_parent = live_arms["contract_lattice_shared_cost_full"]
    single_factor_specs = (
        ("p5_physical_plan", "observational_backend_physical_plan_collection", {"plan_guidance": "physical"}),
        ("p5_goal_first", "goal_first_backward_generation", {"generation_mode": "goal_first"}),
        ("p5_boundary_targeted", "fault_model_targeted_boundary_values", {"boundary_mode": "fault_model_targeted"}),
        ("p5_priority_v2", "interaction_cost_normalized_obligation_priority_v2", {"obligation_priority_mode": "ccs_risk_priority_v2"}),
        ("p5_prefix_localization", "prefix_adaptive_first_divergence_localization", {"localization_mode": "prefix_adaptive"}),
        ("p5_quality_diversity", "semantic_plan_quality_diversity_archive", {"corpus_mode": "semantic_plan_qd"}),
    )
    for arm_id, unique_factor, changes in single_factor_specs:
        arms[arm_id] = replace(
            p5_parent,
            arm_id=arm_id,
            method_version=P5_METHOD_VERSION,
            parent_arm_id="contract_lattice_shared_cost_full",
            unique_factor=unique_factor,
            changed_dimensions=tuple(changes),
            **changes,
        )

    child(
        "p5_physical_goal_first",
        "p5_physical_plan",
        "goal_first_backward_generation",
        method_version=P5_METHOD_VERSION,
        generation_mode="goal_first",
    )
    child(
        "p5_physical_goal_boundary",
        "p5_physical_goal_first",
        "fault_model_targeted_boundary_values",
        method_version=P5_METHOD_VERSION,
        boundary_mode="fault_model_targeted",
    )
    child(
        "p5_physical_goal_boundary_localize",
        "p5_physical_goal_boundary",
        "prefix_adaptive_first_divergence_localization",
        method_version=P5_METHOD_VERSION,
        localization_mode="prefix_adaptive",
    )
    generated_p5 = replace(
        arms["p5_physical_goal_boundary_localize"],
        arm_id="p5_promoted_method",
        method_version=P5_METHOD_VERSION,
        parent_arm_id="p5_physical_goal_boundary_localize",
        unique_factor="semantic_plan_quality_diversity_archive",
        changed_dimensions=("corpus_mode",),
        corpus_mode="semantic_plan_qd",
    )
    _require_same_live_arm(generated_p5, live_arms)
    child(
        "p5_physical_goal_boundary_priority",
        "p5_physical_goal_boundary",
        "interaction_cost_normalized_obligation_priority_v2",
        method_version=P5_METHOD_VERSION,
        obligation_priority_mode="ccs_risk_priority_v2",
    )
    child(
        "p5_physical_goal_boundary_priority_localize",
        "p5_physical_goal_boundary_priority",
        "prefix_adaptive_first_divergence_localization",
        method_version=P5_METHOD_VERSION,
        localization_mode="prefix_adaptive",
    )
    child(
        "p5_full_method",
        "p5_physical_goal_boundary_priority_localize",
        "semantic_plan_quality_diversity_archive",
        method_version=P5_METHOD_VERSION,
        corpus_mode="semantic_plan_qd",
    )
    return arms


def _require_same_live_arm(
    generated: MethodArm,
    live_arms: Mapping[str, MethodArm],
) -> None:
    live = live_arms[generated.arm_id]
    if generated != live:
        raise AssertionError(
            f"lazy research lineage disagrees with live arm: {generated.arm_id}"
        )
