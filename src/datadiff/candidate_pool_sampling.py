from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from datadiff.candidate_burst import (
    NoveltyAwareCandidateBurst,
    candidate_family_evidence,
    row_has_candidate_signal,
)


@dataclass(frozen=True, slots=True)
class CandidatePoolSelection:
    iteration: int
    mode: str
    pool_size: int
    configured_pool_size: int
    minimum_pool_size: int
    candidate_burst_remaining: int
    preserve_seed_stride: bool = False
    compensate_seed_horizon: bool = False

    @property
    def sampled(self) -> bool:
        return self.pool_size < self.configured_pool_size

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "mode": self.mode,
            "sampled": self.sampled,
            "pool_size": self.pool_size,
            "configured_pool_size": self.configured_pool_size,
            "minimum_pool_size": self.minimum_pool_size,
            "candidate_burst_remaining": self.candidate_burst_remaining,
            "preserve_seed_stride": self.preserve_seed_stride,
            "compensate_seed_horizon": self.compensate_seed_horizon,
        }


class AdaptiveCandidatePoolController:
    """Reduce guidance generation cost while retaining periodic full-pool coverage."""

    def __init__(
        self,
        configured_pool_size: int,
        *,
        enabled: bool,
        minimum_pool_size: int = 4,
        full_sweep_interval: int = 12,
        calibration_cases: int = 8,
        candidate_burst_cases: int = 8,
        candidate_burst_novel_only: bool = False,
        preserve_seed_stride: bool = False,
        compensate_seed_horizon: bool = False,
    ) -> None:
        self.configured_pool_size = max(1, int(configured_pool_size or 1))
        self.minimum_pool_size = min(
            self.configured_pool_size,
            max(1, int(minimum_pool_size or 1)),
        )
        self.full_sweep_interval = max(0, int(full_sweep_interval or 0))
        self.calibration_cases = max(0, int(calibration_cases or 0))
        self.candidate_burst_cases = max(0, int(candidate_burst_cases or 0))
        self.candidate_burst_novel_only = bool(candidate_burst_novel_only)
        self.enabled = bool(
            enabled and self.configured_pool_size > self.minimum_pool_size
        )
        self.compensate_seed_horizon = bool(
            compensate_seed_horizon and self.enabled
        )
        self.preserve_seed_stride = bool(
            (preserve_seed_stride or self.compensate_seed_horizon) and self.enabled
        )
        self.iteration = 0
        self._candidate_burst = NoveltyAwareCandidateBurst(
            burst_cases=self.candidate_burst_cases,
            novel_only=self.candidate_burst_novel_only,
        )
        self._selected_candidate_budget = 0
        self._mode_counts: Counter[str] = Counter()
        self._generated_seed_advance = 0
        self._applied_seed_advance = 0
        self._skipped_seed_slots = 0
        self._seed_horizon_compensation_extra_slots = 0

    def select(self) -> CandidatePoolSelection:
        selection_iteration = self.iteration
        if not self.enabled:
            pool_size = self.configured_pool_size
            mode = "fixed_pool"
        elif selection_iteration < self.calibration_cases:
            pool_size = self.configured_pool_size
            mode = "calibration_full_pool"
        elif self._candidate_burst.remaining > 0:
            pool_size = self.configured_pool_size
            mode = "candidate_full_pool"
            self._candidate_burst.consume_case()
        elif (
            self.full_sweep_interval > 0
            and selection_iteration % self.full_sweep_interval == 0
        ):
            pool_size = self.configured_pool_size
            mode = "periodic_full_pool"
        else:
            pool_size = self.minimum_pool_size
            mode = "reduced_pool"

        self.iteration += 1
        self._selected_candidate_budget += pool_size
        self._mode_counts[mode] += 1
        return CandidatePoolSelection(
            iteration=selection_iteration,
            mode=mode,
            pool_size=pool_size,
            configured_pool_size=self.configured_pool_size,
            minimum_pool_size=self.minimum_pool_size,
            candidate_burst_remaining=self._candidate_burst.remaining,
            preserve_seed_stride=self.preserve_seed_stride,
            compensate_seed_horizon=self.compensate_seed_horizon,
        )

    def seed_advance_decision(
        self,
        *,
        generated_seed_advance: int,
        selected_pool_size: int,
    ) -> dict[str, Any]:
        generated = max(0, int(generated_seed_advance))
        selected = max(1, int(selected_pool_size or 1))
        fixed_stride_floor = (
            self.configured_pool_size if self.preserve_seed_stride else 0
        )
        full_pool_equivalent = generated
        if (
            self.compensate_seed_horizon
            and selected < self.configured_pool_size
        ):
            full_pool_equivalent = (
                generated * self.configured_pool_size + selected - 1
            ) // selected
        applied = max(generated, fixed_stride_floor, full_pool_equivalent)
        compatibility_floor = max(generated, fixed_stride_floor)
        compensation_extra = max(0, applied - compatibility_floor)
        return {
            "seed_horizon_policy": (
                "acceptance_compensated"
                if self.compensate_seed_horizon
                else "fixed_full_pool_stride"
                if self.preserve_seed_stride
                else "generated_advance"
            ),
            "generated_seed_advance": generated,
            "fixed_stride_floor": fixed_stride_floor,
            "full_pool_equivalent_seed_advance": full_pool_equivalent,
            "applied_seed_advance": applied,
            "skipped_seed_slots": applied - generated,
            "seed_horizon_compensation_extra_slots": compensation_extra,
        }

    def record_seed_advance(
        self,
        *,
        generated_seed_advance: int,
        applied_seed_advance: int,
        seed_horizon_compensation_extra_slots: int = 0,
    ) -> None:
        generated = max(0, int(generated_seed_advance))
        applied = max(generated, int(applied_seed_advance))
        self._generated_seed_advance += generated
        self._applied_seed_advance += applied
        self._skipped_seed_slots += applied - generated
        self._seed_horizon_compensation_extra_slots += max(
            0,
            int(seed_horizon_compensation_extra_slots),
        )

    def record_outcome(
        self,
        *,
        candidate_detected: bool,
        candidate_families: Iterable[str] = (),
    ) -> dict[str, object]:
        return self._candidate_burst.record(
            policy_enabled=self.enabled,
            candidate_detected=candidate_detected,
            candidate_families=candidate_families,
        )

    def summary(self) -> dict[str, Any]:
        reduced_pool_cases = int(self._mode_counts.get("reduced_pool", 0))
        full_pool_cases = self.iteration - reduced_pool_cases
        return {
            "enabled": self.enabled,
            "configured_pool_size": self.configured_pool_size,
            "minimum_pool_size": self.minimum_pool_size,
            "full_sweep_interval": self.full_sweep_interval,
            "calibration_cases": self.calibration_cases,
            "candidate_burst_cases": self.candidate_burst_cases,
            "iterations": self.iteration,
            **self._candidate_burst.summary(),
            "preserve_seed_stride": self.preserve_seed_stride,
            "compensate_seed_horizon": self.compensate_seed_horizon,
            "generated_seed_advance": self._generated_seed_advance,
            "applied_seed_advance": self._applied_seed_advance,
            "skipped_seed_slots": self._skipped_seed_slots,
            "seed_horizon_compensation_extra_slots": (
                self._seed_horizon_compensation_extra_slots
            ),
            "selected_candidate_budget": self._selected_candidate_budget,
            "mean_pool_size": (
                self._selected_candidate_budget / self.iteration
                if self.iteration
                else 0.0
            ),
            "full_pool_cases": full_pool_cases,
            "reduced_pool_cases": reduced_pool_cases,
            "mode_counts": dict(sorted(self._mode_counts.items())),
        }


def record_candidate_pool_outcome(
    controller: AdaptiveCandidatePoolController,
    *,
    row: dict[str, Any],
    screening_row: dict[str, Any] | None = None,
) -> dict[str, object]:
    full_candidate = row_has_candidate_signal(row)
    screening_candidate = row_has_candidate_signal(screening_row)
    family_evidence = candidate_family_evidence(row)
    decision = controller.record_outcome(
        candidate_detected=bool(full_candidate or screening_candidate),
        candidate_families=family_evidence["confirmed_candidate_families"],
    )
    decision.update(family_evidence)
    decision["screening_observed_candidate_families"] = (
        candidate_family_evidence(screening_row)["observed_candidate_families"]
        if screening_row is not None
        else []
    )
    decision["outcome_source"] = (
        "full_execution_candidate"
        if full_candidate
        else "screening_only_candidate"
        if screening_candidate
        else "no_candidate"
    )
    return decision
