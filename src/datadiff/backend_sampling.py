from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

from datadiff.candidate_burst import (
    NoveltyAwareCandidateBurst,
    candidate_family_evidence,
    row_has_candidate_signal,
)
from datadiff.run_logging import (
    STAGE_PROFILE_KEYS,
    _merge_process_cpu_profile,
    _merge_wall_time_profile,
    _process_cpu_profile_with_total,
    _stage_profile_with_total,
    _wall_time_profile_with_total,
)
from datadiff.execution_accounting import (
    add_execution_profile_backend_work,
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
    execution_profile_cache_hits,
)


def normalize_backend_pair(pair: str) -> str:
    parts = [part.strip() for part in str(pair or "").split("|", 1)]
    if len(parts) != 2 or not parts[0] or not parts[1] or parts[0] == parts[1]:
        return ""
    return "|".join(sorted(parts))


@dataclass(frozen=True, slots=True)
class BackendSelection:
    iteration: int
    mode: str
    active_backends: tuple[str, ...]
    omitted_backends: tuple[str, ...]
    priority_pairs: tuple[str, ...]
    coverage_after: dict[str, object]

    @property
    def sampled(self) -> bool:
        return bool(self.omitted_backends)

    def to_dict(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "mode": self.mode,
            "sampled": self.sampled,
            "active_backends": list(self.active_backends),
            "omitted_backends": list(self.omitted_backends),
            "priority_pairs": list(self.priority_pairs),
            "coverage_after": dict(self.coverage_after),
        }


class CoverageAwareBackendSampler:
    """Balance differential pairs while periodically retaining full-suite checks."""

    def __init__(
        self,
        backends: Iterable[str],
        *,
        enabled: bool,
        sample_size: int = 3,
        full_sweep_interval: int = 16,
        calibration_cases: int = 0,
        candidate_burst_cases: int = 0,
        candidate_burst_novel_only: bool = False,
    ) -> None:
        self.backends = tuple(dict.fromkeys(str(name).strip() for name in backends if str(name).strip()))
        minimum = 2 if len(self.backends) >= 2 else len(self.backends)
        self.sample_size = min(len(self.backends), max(minimum, int(sample_size or minimum)))
        self.full_sweep_interval = max(0, int(full_sweep_interval or 0))
        self.calibration_cases = max(0, int(calibration_cases or 0))
        self.candidate_burst_cases = max(0, int(candidate_burst_cases or 0))
        self.candidate_burst_novel_only = bool(candidate_burst_novel_only)
        self.enabled = bool(enabled and len(self.backends) > self.sample_size)
        self.iteration = 0
        self._candidate_burst = NoveltyAwareCandidateBurst(
            burst_cases=self.candidate_burst_cases,
            novel_only=self.candidate_burst_novel_only,
        )
        self._mode_counts: Counter[str] = Counter()
        self._backend_counts: Counter[str] = Counter()
        self._pair_counts: Counter[str] = Counter()
        self._pairs = tuple(
            normalize_backend_pair(f"{left}|{right}")
            for left, right in combinations(self.backends, 2)
        )

    def select(
        self,
        *,
        priority_pairs: Iterable[str] = (),
        required_backends: Iterable[str] = (),
    ) -> BackendSelection:
        normalized_priority = tuple(
            dict.fromkeys(
                pair
                for pair in (normalize_backend_pair(item) for item in priority_pairs)
                if pair in self._pair_counts or pair in self._pairs
            )
        )
        requested = tuple(
            dict.fromkeys(
                str(name).strip()
                for name in required_backends
                if str(name).strip()
            )
        )
        requested_set = set(requested)
        unknown = sorted(requested_set - set(self.backends))
        if unknown:
            raise ValueError(
                "required backends are outside the configured suite: "
                + ", ".join(unknown)
            )
        normalized_required = tuple(
            backend for backend in self.backends if backend in requested_set
        )
        if requested and len(normalized_required) < min(2, len(self.backends)):
            raise ValueError("registered family backend scope must contain two backends")
        selection_iteration = self.iteration
        if normalized_required:
            active = normalized_required
            mode = "registered_family_scope"
        elif not self.enabled:
            active = self.backends
            mode = "full_suite"
        elif selection_iteration < self.calibration_cases:
            active = self.backends
            mode = "calibration_full_sweep"
        elif self._candidate_burst.remaining > 0:
            active = self.backends
            mode = "candidate_full_sweep"
            self._candidate_burst.consume_case()
        elif (
            self.full_sweep_interval > 0
            and selection_iteration % self.full_sweep_interval == 0
        ):
            active = self.backends
            mode = "full_sweep"
        else:
            active = self._balanced_subset(normalized_priority)
            mode = "coverage_sample"
        self._record(active)
        self._mode_counts[mode] += 1
        self.iteration += 1
        active_set = set(active)
        return BackendSelection(
            iteration=selection_iteration,
            mode=mode,
            active_backends=active,
            omitted_backends=tuple(name for name in self.backends if name not in active_set),
            priority_pairs=normalized_priority,
            coverage_after=self.coverage_summary(),
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

    def coverage_summary(self) -> dict[str, object]:
        backend_values = [self._backend_counts[name] for name in self.backends]
        pair_values = [self._pair_counts[pair] for pair in self._pairs]
        return {
            "iterations": self.iteration,
            "mode_counts": dict(sorted(self._mode_counts.items())),
            **self._candidate_burst.summary(),
            "backend_min": min(backend_values, default=0),
            "backend_max": max(backend_values, default=0),
            "pair_min": min(pair_values, default=0),
            "pair_max": max(pair_values, default=0),
            "backend_counts": {name: self._backend_counts[name] for name in self.backends},
            "pair_counts": {pair: self._pair_counts[pair] for pair in self._pairs},
        }

    def _balanced_subset(self, priority_pairs: tuple[str, ...]) -> tuple[str, ...]:
        priority = set(priority_pairs)
        pair_order = list(self._pairs)
        if pair_order:
            shift = self.iteration % len(pair_order)
            pair_order = pair_order[shift:] + pair_order[:shift]
        seed_pair = min(
            pair_order,
            key=lambda pair: (
                self._pair_counts[pair],
                0 if pair in priority else 1,
            ),
        )
        selected = seed_pair.split("|")
        selected_set = set(selected)
        while len(selected) < self.sample_size:
            remaining = [name for name in self.backends if name not in selected_set]
            candidate = max(
                remaining,
                key=lambda name: (
                    sum(
                        1.0 / (1.0 + self._pair_counts[normalize_backend_pair(f"{name}|{other}")])
                        for other in selected
                    ),
                    sum(
                        normalize_backend_pair(f"{name}|{other}") in priority
                        for other in selected
                    ),
                    1.0 / (1.0 + self._backend_counts[name]),
                    -self.backends.index(name),
                ),
            )
            selected.append(candidate)
            selected_set.add(candidate)
        return tuple(name for name in self.backends if name in selected_set)

    def _record(self, active: tuple[str, ...]) -> None:
        self._backend_counts.update(active)
        self._pair_counts.update(
            normalize_backend_pair(f"{left}|{right}")
            for left, right in combinations(active, 2)
        )


def record_backend_sampling_outcome(
    sampler: CoverageAwareBackendSampler,
    *,
    row: dict[str, Any],
    screening_row: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Keep sampling-policy state transitions outside the orchestration layer."""
    full_candidate = row_has_candidate_signal(row)
    screening_candidate = row_has_candidate_signal(screening_row)
    family_evidence = candidate_family_evidence(row)
    decision = sampler.record_outcome(
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


def backend_sampling_screen_summary(row: dict[str, Any]) -> dict[str, Any]:
    execution_profile = row.get("execution_profile", {})
    return {
        "status": str(row.get("status", "")),
        "candidate_signal": row_has_candidate_signal(row),
        "duration_ms": float(row.get("duration_ms", 0.0) or 0.0),
        "behavior_signature": str(row.get("behavior_signature", "")),
        "discovery_signature": str(row.get("discovery_signature", "")),
        "finding_count": len(row.get("findings", []) or []),
        "candidate_families": sorted(
            {
                f"{finding.get('root_cause', 'unknown')}@{','.join(finding.get('suspicious_backends', []) or [])}"
                for finding in row.get("findings", []) or []
                if finding.get("adjudication", {}).get("countable_as_bug_evidence") is True
            }
        ),
        "stage_profile": dict(row.get("stage_profile", {}) or {}),
        "wall_time_profile": dict(row.get("wall_time_profile", {}) or {}),
        "process_cpu_profile": dict(row.get("process_cpu_profile", {}) or {}),
        "backend_reported_total_ms": execution_profile_backend_reported_ms(execution_profile),
        "backend_calls": execution_profile_backend_calls(execution_profile),
        "execution_cache_hits": execution_profile_cache_hits(execution_profile),
        "execution_reuse": dict(row.get("execution_reuse", {}) or {}),
    }


def attach_backend_sampling_result(
    row: dict[str, Any],
    *,
    selection: BackendSelection,
    configured_backends: list[str],
    screening_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampling = selection.to_dict()
    sampling["configured_backends"] = list(configured_backends)
    sampling["execution_behavior_signature"] = str(row.get("behavior_signature", ""))
    sampling["execution_discovery_signature"] = str(row.get("discovery_signature", ""))
    sampling["confirmation_executed"] = screening_row is not None
    sampling["confirmation_status"] = (
        str(row.get("status", "")) if screening_row is not None else "not_needed"
    )
    sampling["confirmation_candidate_signal"] = bool(
        screening_row is not None and row_has_candidate_signal(row)
    )
    sampling["confirmation_recheck_attempts"] = (
        int((row.get("candidate_recheck", {}) or {}).get("attempts", 0) or 0)
        if screening_row is not None
        else 0
    )
    if screening_row is not None:
        sampling["screening"] = backend_sampling_screen_summary(screening_row)
        combined_profile = _stage_profile_with_total(row.get("stage_profile", {}))
        screening_profile = _stage_profile_with_total(screening_row.get("stage_profile", {}))
        for key in STAGE_PROFILE_KEYS:
            if key != "total_case_wall_ms":
                combined_profile[key] += screening_profile[key]
        combined_profile["total_case_wall_ms"] = sum(
            combined_profile[key]
            for key in STAGE_PROFILE_KEYS
            if key != "total_case_wall_ms"
        )
        row["stage_profile"] = combined_profile
        row["duration_ms"] = combined_profile["total_case_wall_ms"]
        row["wall_time_profile"] = _wall_time_profile_with_total(
            _merge_wall_time_profile(
                row.get("wall_time_profile", {}),
                screening_row.get("wall_time_profile", {}),
            )
        )
        row["process_cpu_profile"] = _process_cpu_profile_with_total(
            _merge_process_cpu_profile(
                row.get("process_cpu_profile", {}),
                screening_row.get("process_cpu_profile", {}),
            )
        )
        execution_profile = dict(row.get("execution_profile", {}) or {})
        screening_execution_profile = screening_row.get("execution_profile", {}) or {}
        screening_backend_ms = execution_profile_backend_reported_ms(
            screening_execution_profile
        )
        screening_backend_calls = execution_profile_backend_calls(
            screening_execution_profile
        )
        screening_cache_hits = execution_profile_cache_hits(
            screening_execution_profile
        )
        execution_profile = add_execution_profile_backend_work(
            execution_profile,
            label="screening",
            additional_ms=screening_backend_ms,
            additional_calls=screening_backend_calls,
            additional_cache_hits=screening_cache_hits,
        )
        row["execution_profile"] = execution_profile
    row["backend_sampling"] = sampling
    sampling["execution_reuse"] = dict(row.get("execution_reuse", {}) or {})
    return row
