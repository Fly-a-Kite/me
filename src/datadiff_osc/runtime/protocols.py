"""Non-executing support plans for the v3 gate and preregistered 2h pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive


@dataclass(frozen=True, slots=True)
class V3GatePlan:
    lane_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    cases_per_run: int
    campaign_rechecks: int
    pipeline_rechecks: int
    source_digest: str
    protocol_digest: str
    execution_authorized: bool = False
    twenty_four_hour_run_authorized: bool = False
    schema_version: str = "osc-post-refactor-v3-gate-plan-v1"

    def __post_init__(self) -> None:
        if len(self.lane_ids) != 11 or len(set(self.lane_ids)) != 11:
            raise ValueError("v3 gate plan requires exactly 11 unique lanes")
        if len(self.seeds) != 2 or len(set(self.seeds)) != 2:
            raise ValueError("v3 gate plan requires exactly two unique seeds")
        if self.cases_per_run != 100:
            raise ValueError("v3 gate plan requires exactly 100 cases per run")
        if self.campaign_rechecks != 3 or self.pipeline_rechecks != 3:
            raise ValueError("v3 candidate rechecks must be frozen at 3/3 and 3/3")
        if not self.source_digest or not self.protocol_digest:
            raise ValueError("v3 gate plan requires source and protocol digests")
        if self.execution_authorized or self.twenty_four_hour_run_authorized:
            raise ValueError("support plan cannot authorize execution or 24h")

    @property
    def planned_runs(self) -> int:
        return len(self.lane_ids) * len(self.seeds)

    @property
    def planned_cases(self) -> int:
        return self.planned_runs * self.cases_per_run

    @property
    def digest(self) -> str:
        return stable_digest("osc-post-refactor-v3-gate-plan", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def build_v3_gate_plan(
    *,
    lane_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    source_digest: str,
    protocol_digest: str,
) -> V3GatePlan:
    return V3GatePlan(
        lane_ids=tuple(lane_ids),
        seeds=tuple(int(seed) for seed in seeds),
        cases_per_run=100,
        campaign_rechecks=3,
        pipeline_rechecks=3,
        source_digest=source_digest,
        protocol_digest=protocol_digest,
        execution_authorized=False,
        twenty_four_hour_run_authorized=False,
    )


@dataclass(frozen=True, slots=True)
class PairedPilotPlan:
    treatment_id: str
    control_id: str
    seed_blocks: tuple[int, ...]
    duration_seconds_per_arm: int
    case_cap_per_arm: int
    cpu_seconds_cap_per_arm: int
    wall_seconds_cap_per_arm: int
    source_digest: str
    environment_digest: str
    target_digest: str
    contract_digest: str
    phase6_gate_report_digest: str
    stopping_rule: str
    primary_metrics: tuple[str, ...]
    execution_authorized: bool = False
    twenty_four_hour_run_authorized: bool = False
    schema_version: str = "osc-preregistered-2h-paired-pilot-plan-v1"

    def __post_init__(self) -> None:
        if not self.treatment_id or not self.control_id or self.treatment_id == self.control_id:
            raise ValueError("paired pilot requires distinct treatment and control")
        if len(self.seed_blocks) < 10 or len(set(self.seed_blocks)) != len(self.seed_blocks):
            raise ValueError("paired pilot requires at least 10 unique seed blocks")
        if self.duration_seconds_per_arm != 7200 or self.wall_seconds_cap_per_arm != 7200:
            raise ValueError("paired pilot wall duration must be frozen at 2h per arm")
        if self.case_cap_per_arm < 1 or self.cpu_seconds_cap_per_arm < 1:
            raise ValueError("paired pilot requires positive equal-arm budgets")
        for value in (
            self.source_digest,
            self.environment_digest,
            self.target_digest,
            self.contract_digest,
            self.phase6_gate_report_digest,
            self.stopping_rule,
        ):
            if not value:
                raise ValueError("paired pilot freeze fields must be non-empty")
        if not self.primary_metrics or len(set(self.primary_metrics)) != len(self.primary_metrics):
            raise ValueError("paired pilot metrics must be non-empty and unique")
        if self.execution_authorized or self.twenty_four_hour_run_authorized:
            raise ValueError("pilot support plan cannot authorize execution or 24h")

    @property
    def digest(self) -> str:
        return stable_digest("osc-preregistered-2h-paired-pilot-plan", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def build_paired_pilot_plan(
    *,
    treatment_id: str,
    control_id: str,
    seed_blocks: tuple[int, ...],
    case_cap_per_arm: int,
    cpu_seconds_cap_per_arm: int,
    source_digest: str,
    environment_digest: str,
    target_digest: str,
    contract_digest: str,
    phase6_gate_report_digest: str,
) -> PairedPilotPlan:
    return PairedPilotPlan(
        treatment_id=treatment_id,
        control_id=control_id,
        seed_blocks=tuple(int(seed) for seed in seed_blocks),
        duration_seconds_per_arm=7200,
        case_cap_per_arm=int(case_cap_per_arm),
        cpu_seconds_cap_per_arm=int(cpu_seconds_cap_per_arm),
        wall_seconds_cap_per_arm=7200,
        source_digest=source_digest,
        environment_digest=environment_digest,
        target_digest=target_digest,
        contract_digest=contract_digest,
        phase6_gate_report_digest=phase6_gate_report_digest,
        stopping_rule="run both arms to the frozen cap; never stop early for favorable outcomes",
        primary_metrics=(
            "strict_recheck_survivors_per_10k_executed_cases",
            "issue_ready_unique_fresh_roots_per_cpu_hour",
            "scheduled_activation",
            "false_positive_rate",
            "wall_and_process_cpu",
        ),
        execution_authorized=False,
        twenty_four_hour_run_authorized=False,
    )
