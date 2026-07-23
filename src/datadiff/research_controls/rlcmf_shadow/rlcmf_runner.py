from __future__ import annotations

from typing import Any, Callable

from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.research_controls.rlcmf_shadow.rlcmf_counterfactual import (
    RLCMFAuditedCaseResult,
    RLCMFCounterfactualExecutor,
)
from datadiff.metamorphic import all_metamorphic_variants, select_metamorphic_variants
from datadiff.research_controls.rlcmf_shadow.rlcmf_factorial import derive_factorial_candidate_row
from datadiff.research_controls.rlcmf_shadow.rlcmf_sentinel import (
    RLCMFSentinelCoordinator,
    SentinelAuditPlan,
)


def _target_specs_for(
    target_specs: list[dict[str, Any]],
    backends: list[str],
) -> list[dict[str, Any]]:
    by_name = {
        str(row.get("name", "")): row
        for row in target_specs
        if isinstance(row, dict) and str(row.get("name", ""))
    }
    return [by_name[name] for name in backends if name in by_name]


def run_rlcmf_audited_loaded_case(
    case: Case,
    *,
    coordinator: RLCMFSentinelCoordinator,
    plans: dict[str, SentinelAuditPlan],
    low_backends: list[str],
    reference_backends: list[str],
    low_config: ExperimentConfig,
    reference_config: ExperimentConfig,
    low_metamorphic_relation_order: list[str] | tuple[str, ...] = (),
    reference_metamorphic_relation_order: list[str] | tuple[str, ...] = (),
    backend_instances: dict[str, Any] | None = None,
    execution_session: Any | None = None,
    environment: dict[str, str] | None = None,
    target_specs: list[dict[str, Any]] | None = None,
    save_low_artifact: bool = False,
    save_audit_artifacts: bool = False,
    enable_factorial_derivation: bool = True,
    run_loaded_case_fn: Callable[..., dict[str, Any]] | None = None,
) -> RLCMFAuditedCaseResult:
    """Run one manifest-planned backend/relation/joint counterfactual case.

    Audit arms intentionally receive fresh backend instances and no reusable
    execution session. This prevents state contamination from masquerading as
    a stable full-fidelity result.
    """

    if run_loaded_case_fn is None:
        from datadiff.runner import run_loaded_case as run_loaded_case_fn

    resolved_targets = list(target_specs or [])

    def run_arm(
        *,
        arm: str,
        backends: list[str],
        config: ExperimentConfig,
        fresh_backend_instances: bool,
    ) -> dict[str, Any]:
        use_reference_relations = arm in {"relation", "joint"}
        relation_order = (
            reference_metamorphic_relation_order
            if use_reference_relations
            else low_metamorphic_relation_order
        )
        return run_loaded_case_fn(
            case,
            backends=backends,
            config=config,
            save_artifact=(
                save_audit_artifacts if arm != "low" else save_low_artifact
            ),
            backend_instances=(None if fresh_backend_instances else backend_instances),
            environment=environment,
            target_specs=_target_specs_for(resolved_targets, backends),
            config_payload=config.to_dict(),
            metamorphic_relation_order=list(relation_order),
            execution_session=(None if fresh_backend_instances else execution_session),
        )

    factorial_deriver = None
    if enable_factorial_derivation:
        registered_metrics = {
            coordinator.event_accumulators[axis].metric
            for axis in plans
            if axis in coordinator.event_accumulators
        }
        if any("confirmed" in metric for metric in registered_metrics):
            raise ValueError(
                "factorial derivation cannot replace fresh confirmed-event audits"
            )
        applicable_variants = list(all_metamorphic_variants(case))
        low_variant_names = [
            variant.name
            for variant in select_metamorphic_variants(
                applicable_variants,
                limit=max(0, int(low_config.metamorphic_variant_limit)),
                relation_order=low_metamorphic_relation_order,
            )
        ]
        reference_variant_names = [
            variant.name
            for variant in select_metamorphic_variants(
                applicable_variants,
                limit=max(0, int(reference_config.metamorphic_variant_limit)),
                relation_order=reference_metamorphic_relation_order,
            )
        ]

        def factorial_deriver(*, axis, joint_row, backends, config):
            return derive_factorial_candidate_row(
                case,
                joint_row,
                backends=backends,
                variant_names=(
                    low_variant_names if axis == "backend" else reference_variant_names
                ),
                config=config,
            )

    executor = RLCMFCounterfactualExecutor(
        coordinator=coordinator,
        run_arm_fn=run_arm,
        factorial_deriver=factorial_deriver,
    )
    result = executor.execute(
        low_backends=low_backends,
        reference_backends=reference_backends,
        low_config=low_config,
        reference_config=reference_config,
        plans=plans,
    )
    result.low_row["rlcmf_counterfactual"] = result.summary()
    return result
