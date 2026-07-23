from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datadiff.method_arms import (
    DEFAULT_METHOD_ARM_ID,
    arm_difference,
    method_arm,
    resolve_method_arm,
)


P_STAGE_METHOD_MATRIX_SCHEMA_VERSION = "icse-fse-p45-method-matrix-v2"
MATRIX_ID = "icse-fse-p45-single-factor-matrix-v2"
CONTROL_ARM_ID = "p45_control_cartesian_full"
P5_BASELINE_ARM_ID = "p5_promoted_shared_cost_full_baseline"


CONTROL_SETTINGS: dict[str, Any] = {
    "comparison_mode": "contract",
    "observation_mode": "lossless",
    "ir_mode": "ccs_ir",
    "obligation_mode": "ccs_guided",
    "obligation_priority_mode": "complete_builder_order",
    "execution_mode": "cartesian",
    "selector_mode": "all",
    "plan_guidance": "disabled",
    "node_budget": None,
    "cache_mode": "disabled",
    "confirmation_mode": "none",
    "parallelism_mode": "sequential",
    "generation_mode": "forward_random",
    "boundary_mode": "disabled",
    "corpus_mode": "legacy_qd",
    "localization_mode": "reducer_only",
}


@dataclass(frozen=True, slots=True)
class PStageMethodArm:
    arm_id: str
    parent_arm_id: str
    stage: str
    factor: str
    headline_treatment: bool
    settings: dict[str, Any]
    implementation_status: str
    core_method_arm: str = ""
    core_overrides: dict[str, Any] | None = None
    cli_flags: tuple[str, ...] = ()
    implementation_stage: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "arm_id": self.arm_id,
            "parent_arm_id": self.parent_arm_id,
            "stage": self.stage,
            "factor": self.factor,
            "headline_treatment": self.headline_treatment,
            "settings": dict(self.settings),
            "implementation_status": self.implementation_status,
            "core_method_arm": self.core_method_arm,
            "core_overrides": dict(self.core_overrides or {}),
            "cli_flags": list(self.cli_flags),
            "implementation_stage": self.implementation_stage,
            "note": self.note,
        }
        payload["changed_dimensions"] = list(changed_dimensions(self))
        payload["command"] = arm_command(self)
        payload["digest"] = "p45-arm-" + canonical_sha256(payload)
        return payload


def _settings(**changes: Any) -> dict[str, Any]:
    payload = dict(CONTROL_SETTINGS)
    payload.update(changes)
    return payload


_ARMS: tuple[PStageMethodArm, ...] = (
    PStageMethodArm(
        arm_id=CONTROL_ARM_ID,
        parent_arm_id="",
        stage="P3.5/P4/P5 control",
        factor="control",
        headline_treatment=False,
        settings=_settings(),
        implementation_status="available",
        core_method_arm="contract_ccs_obligations_cartesian",
        cli_flags=("--disable-parallel-backend-execution",),
        note="Complete obligations, complete builder order, Cartesian/full backend execution.",
    ),
    PStageMethodArm(
        arm_id="p4_lattice_static_cache_off_bridge",
        parent_arm_id=CONTROL_ARM_ID,
        stage="P4 bridge",
        factor="execution_lattice_bridge",
        headline_treatment=False,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="static",
            node_budget=3.0,
            cache_mode="disabled",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_static",
        core_overrides={"cache_mode": "disabled"},
        cli_flags=("--disable-parallel-backend-execution",),
        note=(
            "Non-headline bridge used only to establish a common cache-off lattice "
            "parent. Its coupled execution/selector/budget transition is never reported "
            "as an isolated treatment effect."
        ),
    ),
    PStageMethodArm(
        arm_id="p4_cache_lattice_on",
        parent_arm_id="p4_lattice_static_cache_off_bridge",
        stage="P4",
        factor="cache_mode",
        headline_treatment=True,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="static",
            node_budget=3.0,
            cache_mode="lattice",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_static",
        cli_flags=("--disable-parallel-backend-execution",),
        note="Cache-on treatment; parent is byte-for-byte the same method except cache_mode.",
    ),
    PStageMethodArm(
        arm_id="p4_selector_shared_cost",
        parent_arm_id="p4_lattice_static_cache_off_bridge",
        stage="P4",
        factor="selector_mode",
        headline_treatment=True,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="disabled",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_static",
        core_overrides={"selector_mode": "shared_cost", "cache_mode": "disabled"},
        cli_flags=("--disable-parallel-backend-execution",),
    ),
    PStageMethodArm(
        arm_id="p4_selector_seeded_random",
        parent_arm_id="p4_lattice_static_cache_off_bridge",
        stage="P4",
        factor="selector_mode",
        headline_treatment=True,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="random",
            node_budget=3.0,
            cache_mode="disabled",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_static",
        core_overrides={"selector_mode": "random", "cache_mode": "disabled"},
        cli_flags=("--disable-parallel-backend-execution",),
    ),
    PStageMethodArm(
        arm_id="p4_selector_plan_unweighted",
        parent_arm_id="p4_lattice_static_cache_off_bridge",
        stage="P4/P5 bridge",
        factor="selector_mode",
        headline_treatment=True,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="plan",
            plan_guidance="disabled",
            node_budget=3.0,
            cache_mode="disabled",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_static",
        core_overrides={"selector_mode": "plan", "cache_mode": "disabled"},
        cli_flags=("--disable-parallel-backend-execution",),
        note="Plan selector without semantic fingerprint input; parent for guidance-only arms.",
    ),
    PStageMethodArm(
        arm_id="p5_semantic_plan_guidance",
        parent_arm_id="p4_selector_plan_unweighted",
        stage="P5",
        factor="plan_guidance",
        headline_treatment=True,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="plan",
            plan_guidance="semantic",
            node_budget=3.0,
            cache_mode="disabled",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_plan",
        core_overrides={"cache_mode": "disabled"},
        cli_flags=("--disable-parallel-backend-execution",),
        note=(
            "Frozen negative control. P4.8-B rejected this semantic selector; P5 does "
            "not tune or use it as the parent of physical-plan work."
        ),
    ),
    PStageMethodArm(
        arm_id=P5_BASELINE_ARM_ID,
        parent_arm_id="p4_lattice_static_cache_off_bridge",
        stage="P4.8-B/P5 bridge",
        factor="promoted_execution_baseline",
        headline_treatment=False,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="contract_lattice_shared_cost_full",
        cli_flags=("--disable-parallel-backend-execution",),
        note=(
            "Frozen P4.8-B production baseline. Its multi-dimension bridge is not "
            "reported as a single treatment effect."
        ),
    ),
    PStageMethodArm(
        arm_id="p5_physical_plan_guidance",
        parent_arm_id=P5_BASELINE_ARM_ID,
        stage="P5",
        factor="plan_guidance",
        headline_treatment=True,
        settings=_settings(
            execution_mode="lattice",
            selector_mode="shared_cost",
            plan_guidance="physical",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="p5_physical_plan",
        cli_flags=("--disable-parallel-backend-execution",),
        implementation_stage="P5.1",
        note=(
            "Observational physical-plan collection on the promoted shared-cost "
            "baseline; it does not reuse the rejected semantic-plan selector."
        ),
    ),
    PStageMethodArm(
        arm_id="p2_priority_v1_negative_control",
        parent_arm_id=CONTROL_ARM_ID,
        stage="P2/P5 control",
        factor="obligation_priority_mode",
        headline_treatment=True,
        settings=_settings(obligation_priority_mode="ccs_risk_priority"),
        implementation_status="available_negative_control",
        core_method_arm="contract_ccs_risk_obligations_cartesian",
        cli_flags=("--disable-parallel-backend-execution",),
        note="Frozen negative control: confirmed-root delta was zero on the 500x5 holdout.",
    ),
    PStageMethodArm(
        arm_id="p5_priority_v2",
        parent_arm_id=P5_BASELINE_ARM_ID,
        stage="P5",
        factor="obligation_priority_mode",
        headline_treatment=True,
        settings=_settings(
            obligation_priority_mode="ccs_risk_priority_v2",
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="p5_priority_v2",
        cli_flags=("--disable-parallel-backend-execution",),
        implementation_stage="P5.5",
        note=(
            "Frozen development-only coefficients score semantic interactions and "
            "prospective plan transitions per estimated execution cost; current full "
            "outcomes and the final holdout are explicitly excluded."
        ),
    ),
    PStageMethodArm(
        arm_id="p5_goal_first_generation",
        parent_arm_id=P5_BASELINE_ARM_ID,
        stage="P5",
        factor="generation_mode",
        headline_treatment=True,
        settings=_settings(
            generation_mode="goal_first",
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="p5_goal_first",
        cli_flags=("--disable-parallel-backend-execution",),
        implementation_stage="P5.3",
        note=(
            "Goal-first selection precedes construction and records backward "
            "preconditions, constructibility, validity, and fallback reasons."
        ),
    ),
    PStageMethodArm(
        arm_id="p5_boundary_targeted_values",
        parent_arm_id=P5_BASELINE_ARM_ID,
        stage="P5",
        factor="boundary_mode",
        headline_treatment=True,
        settings=_settings(
            boundary_mode="fault_model_targeted",
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="p5_boundary_targeted",
        cli_flags=("--disable-parallel-backend-execution",),
        implementation_stage="P5.4",
        note=(
            "Versioned boundary profiles are selected only when logical type, "
            "operation, and fault-model applicability predicates hold."
        ),
    ),
    PStageMethodArm(
        arm_id="p5_prefix_adaptive_localization",
        parent_arm_id=P5_BASELINE_ARM_ID,
        stage="P5",
        factor="localization_mode",
        headline_treatment=True,
        settings=_settings(
            localization_mode="prefix_adaptive",
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="p5_prefix_localization",
        cli_flags=("--disable-parallel-backend-execution",),
        implementation_stage="P5.6",
        note=(
            "After a full-program finding, adaptive prefix execution records a "
            "prefix-result DAG, physical-plan transitions, culprit ranking, and "
            "additive backend-call/wall/process-CPU cost."
        ),
    ),
    PStageMethodArm(
        arm_id="p5_quality_diversity_corpus",
        parent_arm_id=P5_BASELINE_ARM_ID,
        stage="P5",
        factor="corpus_mode",
        headline_treatment=True,
        settings=_settings(
            corpus_mode="semantic_plan_qd",
            execution_mode="lattice",
            selector_mode="shared_cost",
            node_budget=3.0,
            cache_mode="lattice",
            confirmation_mode="full",
        ),
        implementation_status="available",
        core_method_arm="p5_quality_diversity",
        cli_flags=("--disable-parallel-backend-execution",),
        implementation_stage="P5.7",
        note=(
            "Semantic/physical interaction, plan, layout, and cold-stratum axes "
            "drive a QD archive with explicit cold-debt protection and a nonzero "
            "saturated-family audit floor."
        ),
    ),
)


def p_stage_method_arms() -> tuple[PStageMethodArm, ...]:
    return _ARMS


def p_stage_method_arm(arm_id: str) -> PStageMethodArm:
    for arm in _ARMS:
        if arm.arm_id == arm_id:
            return arm
    allowed = ", ".join(arm.arm_id for arm in _ARMS)
    raise ValueError(f"unknown P-stage method arm '{arm_id}'; expected one of: {allowed}")


def changed_dimensions(arm: PStageMethodArm) -> tuple[str, ...]:
    if not arm.parent_arm_id:
        return ()
    parent = p_stage_method_arm(arm.parent_arm_id)
    keys = sorted(set(parent.settings) | set(arm.settings))
    return tuple(key for key in keys if parent.settings.get(key) != arm.settings.get(key))


def arm_command(arm: PStageMethodArm) -> str:
    if arm.implementation_status.startswith("planned"):
        return (
            ".venv/bin/python scripts/validate_p_stage_method_matrix.py "
            f"--arm {arm.arm_id} --require-planned"
        )
    overrides = dict(arm.core_overrides or {})
    registered_control = {
        ("contract_lattice_static", (("cache_mode", "disabled"),)): (
            "p4_lattice_static_cache_off"
        ),
        (
            "contract_lattice_static",
            (("cache_mode", "disabled"), ("selector_mode", "shared_cost")),
        ): "p4_lattice_shared_cost_cache_off",
        (
            "contract_lattice_static",
            (("cache_mode", "disabled"), ("selector_mode", "random")),
        ): "p4_lattice_random_cache_off",
        (
            "contract_lattice_static",
            (("cache_mode", "disabled"), ("selector_mode", "plan")),
        ): "p4_lattice_plan_unguided_cache_off",
        ("contract_lattice_plan", (("cache_mode", "disabled"),)): (
            "p4_lattice_plan_cache_off"
        ),
    }.get(
        (
            arm.core_method_arm,
            tuple(sorted(overrides.items())),
        )
    )
    selected_arm = registered_control or arm.core_method_arm
    arm_flag = (
        "--method-arm"
        if selected_arm in {DEFAULT_METHOD_ARM_ID, "contract_lattice_shared_cost_full", "p5_promoted_method"}
        else "--research-control-arm"
    )
    pieces = [
        ".venv/bin/datadiff",
        "fuzz",
        arm_flag,
        selected_arm,
    ]
    if overrides and registered_control is None:
        raise ValueError(
            f"unregistered control combination for {arm.arm_id}: {overrides}"
        )
    pieces.extend(arm.cli_flags)
    pieces.extend(
        (
            "--seed",
            "1001",
            "--cases",
            "24",
            "--backends",
            "pandas,polars,duckdb,pyarrow,datafusion",
            "--disable-artifact",
        )
    )
    return " ".join(pieces)


def build_p_stage_method_matrix() -> dict[str, Any]:
    validation = validate_p_stage_method_matrix()
    arms = [arm.to_dict() for arm in _ARMS]
    payload: dict[str, Any] = {
        "schema_version": P_STAGE_METHOD_MATRIX_SCHEMA_VERSION,
        "matrix_id": MATRIX_ID,
        "control_arm_id": CONTROL_ARM_ID,
        # This is a frozen P3-P5 matrix. Its historical live default must not be
        # rewritten when the active repository advances to P8.
        "live_default_method_arm": "p5_promoted_method",
        "live_default_method_manifest": method_arm("p5_promoted_method").to_dict(),
        "comparison_block": {
            "case_source": "frozen corpus or preregistered seed derivation",
            "case_order": "identical",
            "backends": ["pandas", "polars", "duckdb", "pyarrow", "datafusion"],
            "backend_order": "identical",
            "target_versions": "identical",
            "environment_digest": "identical",
            "process_cpu_budget": "identical within each paired experiment",
            "stopping_rule": "identical and outcome-blind",
        },
        "arms": arms,
        "parent_graph": [
            {"parent": arm.parent_arm_id, "child": arm.arm_id}
            for arm in _ARMS
            if arm.parent_arm_id
        ],
        "summary": validation["summary"],
        "claim_policy": {
            "bridge_effect_claim_allowed": False,
            "planned_arm_experiment_allowed": False,
            "negative_priority_v1_may_be_default": False,
            "headline_effect_requires_single_changed_dimension": True,
        },
    }
    payload["matrix_digest"] = "p45-matrix-" + canonical_sha256(payload)
    return payload


def validate_p_stage_method_matrix() -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    ids = [arm.arm_id for arm in _ARMS]
    if len(ids) != len(set(ids)):
        errors.append("duplicate arm IDs")
    if ids.count(CONTROL_ARM_ID) != 1:
        errors.append("matrix must contain exactly one control arm")
    seen: set[str] = set()
    available = 0
    planned = 0
    headline = 0
    for arm in _ARMS:
        if arm.parent_arm_id and arm.parent_arm_id not in seen:
            errors.append(f"{arm.arm_id}: parent is missing or appears after child")
        changed = changed_dimensions(arm)
        if arm.headline_treatment:
            headline += 1
            if changed != (arm.factor,):
                errors.append(
                    f"{arm.arm_id}: headline factor {arm.factor} differs from {list(changed)}"
                )
        elif arm.parent_arm_id and not changed:
            errors.append(f"{arm.arm_id}: bridge changes no dimensions")
        if set(arm.settings) != set(CONTROL_SETTINGS):
            errors.append(f"{arm.arm_id}: settings dimension set differs from control")
        if arm.implementation_status.startswith("planned"):
            planned += 1
            if arm.core_method_arm or arm.core_overrides:
                errors.append(f"{arm.arm_id}: planned arm must not pretend to have a core binding")
        else:
            available += 1
            _validate_core_binding(arm, errors)
        if not arm_command(arm):
            errors.append(f"{arm.arm_id}: missing command")
        seen.add(arm.arm_id)

    p4_ids = {
        "p4_lattice_static_cache_off_bridge",
        "p4_cache_lattice_on",
        "p4_selector_shared_cost",
        "p4_selector_seeded_random",
        "p4_selector_plan_unweighted",
    }
    p4_ready = all(
        p_stage_method_arm(arm_id).implementation_status == "available"
        for arm_id in p4_ids
    )
    if planned:
        warnings.append(
            "Quality-diversity corpus mode remains frozen and fails closed until its P5 implementation step."
        )
    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "arm_count": len(_ARMS),
            "headline_treatment_count": headline,
            "available_binding_count": available,
            "planned_fail_closed_count": planned,
            "all_headline_arms_single_factor": not any(
                "headline factor" in error for error in errors
            ),
            "p4_matrix_ready": p4_ready and not errors,
            "p4_start_ready": p4_ready and not errors,
            "p5_matrix_names_and_parents_frozen": True,
            "p3_5_freeze_complete": not errors,
            "all_treatments_implementation_bound": planned == 0,
            "p3_5_exit_ready": not errors and p4_ready and planned == 0,
        },
    }


def write_p_stage_method_matrix(path: Path) -> dict[str, Any]:
    payload = build_p_stage_method_matrix()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_core_binding(arm: PStageMethodArm, errors: list[str]) -> None:
    if not arm.core_method_arm:
        errors.append(f"{arm.arm_id}: available arm has no core method binding")
        return
    try:
        resolution = resolve_method_arm(arm.core_method_arm, arm.core_overrides or {})
    except (TypeError, ValueError) as exc:
        errors.append(f"{arm.arm_id}: invalid core binding: {exc}")
        return
    effective = resolution.effective_arm
    expected_core = {
        "comparison_mode": arm.settings["comparison_mode"],
        "observation_mode": arm.settings["observation_mode"],
        "ir_mode": arm.settings["ir_mode"],
        "obligation_mode": arm.settings["obligation_mode"],
        "obligation_priority_mode": arm.settings["obligation_priority_mode"],
        "execution_mode": arm.settings["execution_mode"],
        "selector_mode": arm.settings["selector_mode"],
        "node_budget": arm.settings["node_budget"],
        "cache_mode": arm.settings["cache_mode"],
        "confirmation_mode": arm.settings["confirmation_mode"],
        "generation_mode": arm.settings["generation_mode"],
        "boundary_mode": arm.settings["boundary_mode"],
        "corpus_mode": arm.settings["corpus_mode"],
        "localization_mode": arm.settings["localization_mode"],
    }
    observed_core = {
        key: getattr(effective, key)
        for key in expected_core
    }
    expected_plan = arm.settings["plan_guidance"]
    observed_core["plan_guidance"] = effective.plan_guidance
    expected_core["plan_guidance"] = expected_plan
    if observed_core != expected_core:
        errors.append(
            f"{arm.arm_id}: core binding mismatch expected={expected_core} observed={observed_core}"
        )
    if "--disable-parallel-backend-execution" not in arm.cli_flags:
        errors.append(f"{arm.arm_id}: comparison control must freeze sequential parallelism")


def core_binding_difference(left_id: str, right_id: str) -> dict[str, tuple[Any, Any]]:
    left = p_stage_method_arm(left_id)
    right = p_stage_method_arm(right_id)
    if not left.core_method_arm or not right.core_method_arm:
        return {}
    return arm_difference(
        resolve_method_arm(left.core_method_arm, left.core_overrides or {}).effective_arm,
        resolve_method_arm(right.core_method_arm, right.core_overrides or {}).effective_arm,
    )
