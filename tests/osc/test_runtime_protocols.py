from __future__ import annotations

import pytest

from datadiff_osc.runtime.protocols import (
    build_paired_pilot_plan,
    build_v3_gate_plan,
)


def test_v3_plan_is_exactly_22_runs_2200_cases_and_never_authorizes_execution():
    plan = build_v3_gate_plan(
        lane_ids=tuple(f"lane-{index}" for index in range(11)),
        seeds=(40000001, 40000002),
        source_digest="source-1",
        protocol_digest="protocol-1",
    )
    assert plan.planned_runs == 22
    assert plan.planned_cases == 2200
    assert plan.campaign_rechecks == plan.pipeline_rechecks == 3
    assert plan.execution_authorized is False
    assert plan.twenty_four_hour_run_authorized is False
    assert plan.digest == build_v3_gate_plan(
        lane_ids=plan.lane_ids,
        seeds=plan.seeds,
        source_digest="source-1",
        protocol_digest="protocol-1",
    ).digest


def test_v3_plan_rejects_changed_hard_denominators():
    with pytest.raises(ValueError, match="11 unique lanes"):
        build_v3_gate_plan(
            lane_ids=("lane",),
            seeds=(1, 2),
            source_digest="source",
            protocol_digest="protocol",
        )
    with pytest.raises(ValueError, match="two unique seeds"):
        build_v3_gate_plan(
            lane_ids=tuple(f"lane-{index}" for index in range(11)),
            seeds=(1, 1),
            source_digest="source",
            protocol_digest="protocol",
        )


def _pilot(seed_blocks=tuple(range(10))):
    return build_paired_pilot_plan(
        treatment_id="full-osc",
        control_id="strongest-internal-baseline",
        seed_blocks=seed_blocks,
        case_cap_per_arm=10000,
        cpu_seconds_cap_per_arm=7200,
        source_digest="source-1",
        environment_digest="environment-1",
        target_digest="target-1",
        contract_digest="contract-1",
        phase6_gate_report_digest="phase6-gates-1",
    )


def test_pilot_plan_freezes_paired_e4_design_but_cannot_run_it():
    plan = _pilot()
    assert len(plan.seed_blocks) == 10
    assert plan.duration_seconds_per_arm == 7200
    assert plan.wall_seconds_cap_per_arm == 7200
    assert plan.execution_authorized is False
    assert plan.twenty_four_hour_run_authorized is False
    assert "never stop early" in plan.stopping_rule
    assert plan.treatment_id != plan.control_id


def test_pilot_plan_rejects_too_few_or_duplicate_seed_blocks():
    with pytest.raises(ValueError, match="at least 10 unique"):
        _pilot(tuple(range(9)))
    with pytest.raises(ValueError, match="at least 10 unique"):
        _pilot((0, 1, 2, 3, 4, 5, 6, 7, 8, 8))
