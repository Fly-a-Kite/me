from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datadiff.finding_outcomes import row_has_rewardable_new_behavior

STAGE_TIMING_KEYS: tuple[str, ...] = (
    "generate_mutate_ms",
    "backend_execution_ms",
    "normalize_ms",
    "oracle_classification_ms",
    "scheduler_feedback_ms",
    "logging_artifact_ms",
    "total_case_wall_ms",
)


def _dict_payload(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = []
    return tuple(str(item) for item in items if str(item))


@dataclass(frozen=True, slots=True)
class FuzzBudget:
    cases: int | None
    duration_s: float | None
    started: float

    @classmethod
    def start(
        cls,
        *,
        cases: int | None,
        duration_s: float | None,
        now: float,
    ) -> "FuzzBudget":
        if cases is None and duration_s is None:
            cases = 100
        return cls(cases=cases, duration_s=duration_s, started=now)

    def elapsed_s(self, now: float) -> float:
        return max(0.0, float(now) - self.started)

    def should_start_iteration(self, *, executed: int, now: float) -> bool:
        if self.cases is not None and executed >= self.cases:
            return False
        if self.duration_s is not None and executed > 0 and self.elapsed_s(now) >= self.duration_s:
            return False
        return True

    def should_stop_after_iteration(self, *, now: float) -> bool:
        return self.duration_s is not None and self.elapsed_s(now) >= self.duration_s


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    run_file: Path
    case_log_file: Path | None
    checkpoint_file: Path | None

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        runs_dir: Path,
        corpus_dir: Path,
        compress_run_log: bool,
        save_cases: bool,
        case_log_file: Path | None,
        checkpoint_interval_s: float | None,
    ) -> "RunPaths":
        run_suffix = ".jsonl.gz" if compress_run_log else ".jsonl"
        run_file = runs_dir / f"{run_id}{run_suffix}"
        resolved_save_cases = bool(save_cases or case_log_file is not None)
        resolved_case_log_file = case_log_file
        if resolved_save_cases and resolved_case_log_file is None:
            resolved_case_log_file = corpus_dir / "generated" / f"{run_id}.cases.jsonl"
        checkpoint_file = runs_dir / f"{run_id}.checkpoint.json" if checkpoint_interval_s is not None else None
        return cls(
            run_id=run_id,
            run_file=run_file,
            case_log_file=resolved_case_log_file,
            checkpoint_file=checkpoint_file,
        )


@dataclass(slots=True)
class IntervalGate:
    interval_s: float | None
    last_fire_s: float

    def due(self, now: float) -> bool:
        return self.interval_s is not None and (float(now) - self.last_fire_s) >= self.interval_s

    def mark(self, now: float) -> None:
        self.last_fire_s = float(now)


@dataclass(slots=True)
class RunCounters:
    known_saturated_bug_families: tuple[str, ...] = ()
    executed: int = 0
    findings: int = 0
    new_behavior_cases: int = 0
    signal_new_behavior_cases: int = 0
    saved_artifacts: int = 0
    preflight_repaired_cases: int = 0
    preflight_fallback_cases: int = 0
    preflight_invalid_cases: int = 0
    replay_filtered_candidates: int = 0
    replay_fallback_candidates: int = 0
    saturated_family_filtered_candidates: int = 0
    saturated_family_fallback_candidates: int = 0
    quality_oracles: dict[str, int] = field(default_factory=dict)
    case_iteration_failures: int = 0
    checkpoint_write_failures: int = 0
    last_case_iteration_error: str = ""

    def preflight_summary(self) -> dict[str, int]:
        return {
            "repaired_cases": self.preflight_repaired_cases,
            "fallback_cases": self.preflight_fallback_cases,
            "invalid_cases": self.preflight_invalid_cases,
        }

    def replay_filter_summary(self, *, enabled: bool) -> dict[str, int | bool]:
        return {
            "enabled": bool(enabled),
            "filtered_candidates": self.replay_filtered_candidates,
            "fallback_candidates": self.replay_fallback_candidates,
        }

    def family_saturation_filter_summary(self, *, enabled: bool) -> dict[str, int | bool]:
        return {
            "enabled": bool(enabled),
            "filtered_candidates": self.saturated_family_filtered_candidates,
            "fallback_candidates": self.saturated_family_fallback_candidates,
        }

    def record_quality_oracle(self, oracle: dict[str, Any]) -> None:
        key = f"{oracle['name']}:{oracle['verdict']}"
        self.quality_oracles[key] = self.quality_oracles.get(key, 0) + 1

    def record_completed_case(
        self,
        *,
        row: dict[str, Any],
        preflight: dict[str, Any],
    ) -> None:
        self.preflight_repaired_cases += int(bool(preflight.get("repaired", False)))
        self.preflight_fallback_cases += int(bool(preflight.get("fallback_used", False)))
        self.preflight_invalid_cases += int(not bool(preflight.get("valid", True)))
        self.findings += len(row.get("findings", []) or [])
        self.new_behavior_cases += int(bool(row.get("is_new_behavior", False)))
        self.signal_new_behavior_cases += int(
            row_has_rewardable_new_behavior(
                row,
                known_saturated_bug_families=self.known_saturated_bug_families,
            )
        )
        self.executed += 1


@dataclass(frozen=True, slots=True)
class StageTimings:
    generate_mutate_ms: float = 0.0
    backend_execution_ms: float = 0.0
    normalize_ms: float = 0.0
    oracle_classification_ms: float = 0.0
    scheduler_feedback_ms: float = 0.0
    logging_artifact_ms: float = 0.0
    total_case_wall_ms: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "StageTimings":
        payload = data if isinstance(data, dict) else {}
        values = {
            key: max(0.0, float(payload.get(key, 0.0) or 0.0))
            for key in STAGE_TIMING_KEYS
        }
        if not values["total_case_wall_ms"]:
            values["total_case_wall_ms"] = sum(
                values[key]
                for key in STAGE_TIMING_KEYS
                if key != "total_case_wall_ms"
            )
        return cls(**values)

    def to_dict(self) -> dict[str, float]:
        return {key: float(getattr(self, key)) for key in STAGE_TIMING_KEYS}


@dataclass(frozen=True, slots=True)
class FuzzIteration:
    run_id: str = ""
    case_index: int = 0
    case_id: str = ""
    seed: int = 0
    candidate_seed_start: int = 0
    candidate_pool_size: int = 1
    candidate_source: str = "generated"
    status: str = "unknown"
    finding_count: int = 0
    is_new_behavior: bool = False
    signal_new_behavior: bool = False
    selected_generator_profile: str = ""
    selected_semantic_objective: str = ""
    selected_metamorphic_relation: str = ""
    selected_version_pair: str = ""
    backend_pair_priority: tuple[str, ...] = ()
    stage_timings: StageTimings = field(default_factory=StageTimings)
    execution_profile: dict[str, Any] = field(default_factory=dict)
    guidance: dict[str, Any] = field(default_factory=dict)
    preflight: dict[str, Any] = field(default_factory=dict)
    operation_combo: dict[str, Any] = field(default_factory=dict)
    reward: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any], *, run_id: str = "") -> "FuzzIteration":
        case = _dict_payload(row.get("case", {}))
        findings = row.get("findings", []) if isinstance(row.get("findings", []), list) else []
        reward = {
            "source_reward": row.get("source_reward"),
            "stored_in_feedback_corpus": bool(row.get("stored_in_feedback_corpus", False)),
            "feedback_eligible": bool(row.get("feedback_eligible", False)),
            "feedback_skip_reason": str(row.get("feedback_skip_reason", "") or ""),
        }
        return cls(
            run_id=str(run_id or row.get("run_id", "") or ""),
            case_index=max(0, int(row.get("case_index", 0) or 0)),
            case_id=str(case.get("case_id", row.get("case_id", "")) or ""),
            seed=int(case.get("seed", row.get("seed", 0)) or 0),
            candidate_seed_start=int(row.get("candidate_seed_start", 0) or 0),
            candidate_pool_size=max(1, int(row.get("candidate_pool_size", 1) or 1)),
            candidate_source=str(row.get("candidate_source", "generated") or "generated"),
            status=str(row.get("status", "unknown") or "unknown"),
            finding_count=len(findings),
            is_new_behavior=bool(row.get("is_new_behavior", False)),
            signal_new_behavior=bool(row.get("signal_new_behavior", False)),
            selected_generator_profile=str(row.get("selected_generator_profile", "") or ""),
            selected_semantic_objective=str(row.get("selected_semantic_objective", "") or ""),
            selected_metamorphic_relation=str(row.get("selected_metamorphic_relation", "") or ""),
            selected_version_pair=str(row.get("selected_version_pair", "") or ""),
            backend_pair_priority=_string_tuple(row.get("backend_pair_priority", [])),
            stage_timings=StageTimings.from_mapping(row.get("stage_profile", {})),
            execution_profile=_dict_payload(row.get("execution_profile", {})),
            guidance=_dict_payload(row.get("guidance", {})),
            preflight=_dict_payload(row.get("preflight", {})),
            operation_combo=_dict_payload(row.get("operation_combo", {})),
            reward=reward,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "case_index": self.case_index,
            "case_id": self.case_id,
            "seed": self.seed,
            "candidate_seed_start": self.candidate_seed_start,
            "candidate_pool_size": self.candidate_pool_size,
            "candidate_source": self.candidate_source,
            "status": self.status,
            "finding_count": self.finding_count,
            "is_new_behavior": self.is_new_behavior,
            "signal_new_behavior": self.signal_new_behavior,
            "selected_generator_profile": self.selected_generator_profile,
            "selected_semantic_objective": self.selected_semantic_objective,
            "selected_metamorphic_relation": self.selected_metamorphic_relation,
            "selected_version_pair": self.selected_version_pair,
            "backend_pair_priority": list(self.backend_pair_priority),
            "stage_timings": self.stage_timings.to_dict(),
            "execution_profile": dict(self.execution_profile),
            "guidance": dict(self.guidance),
            "preflight": dict(self.preflight),
            "operation_combo": dict(self.operation_combo),
            "reward": dict(self.reward),
        }
