"""Deterministic case coordinator for multi-target experiments.

This is intentionally independent of the P7/RLCMF shadow controller.  It
provides the live scheduler's small, auditable decision surface: explicit
source quotas, coverage-debt pressure, resource costs and causal-root
saturation are all visible in one utility value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from datadiff.experiment_manifest import stable_digest
from datadiff.semantic_novelty import (
    SemanticNoveltyDescriptor,
    SemanticNoveltyLedger,
    SemanticNoveltyPreview,
)


class CandidateSource(str, Enum):
    FRESH_GRAMMAR = "fresh_grammar_generation"
    FEEDBACK_MUTATION = "feedback_mutation"
    SEMANTIC_METAMORPHIC = "semantic_metamorphic_mutation"
    HISTORICAL_REPLAY = "historical_replay"
    KNOWN_REGRESSION = "known_regression_replay"
    SEEDED_SENSITIVITY = "seeded_fault_sensitivity"


class FamilyLifecycle(str, Enum):
    NOVEL = "novel"
    CANDIDATE = "candidate"
    LOCALLY_CONFIRMED = "locally_confirmed"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    CONFIRMED = "confirmed"
    FIXED = "fixed"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    SATURATED = "saturated"

    @property
    def fresh_reward_eligible(self) -> bool:
        return self in {FamilyLifecycle.NOVEL, FamilyLifecycle.CANDIDATE, FamilyLifecycle.LOCALLY_CONFIRMED}


class CapabilityStatus(str, Enum):
    WITNESSED = "witnessed"
    UNSUPPORTED_WITH_EVIDENCE = "unsupported-with-evidence"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Utility:
    coverage_debt: float = 0.0
    novelty_upper_bound: float = 0.0
    diversity_gain: float = 0.0
    predicted_cpu_cost: float = 0.0
    predicted_io_cost: float = 0.0
    known_root_saturation_penalty: float = 0.0

    @property
    def score(self) -> float:
        return (
            max(0.0, self.coverage_debt)
            + max(0.0, self.novelty_upper_bound)
            + max(0.0, self.diversity_gain)
            - max(0.0, self.predicted_cpu_cost)
            - max(0.0, self.predicted_io_cost)
            - max(0.0, self.known_root_saturation_penalty)
        )

    def to_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["score"] = self.score
        return payload


@dataclass(frozen=True, slots=True)
class ScheduledCase:
    case_id: str
    case_index: int
    source: CandidateSource
    derived_seed: int
    target_shard: str
    capability_units: tuple[str, ...] = ()
    utility: Utility = Utility()
    causal_signature: str = ""
    semantic_novelty: SemanticNoveltyDescriptor | None = None
    semantic_novelty_preview: SemanticNoveltyPreview | None = None
    family_lifecycle: FamilyLifecycle = FamilyLifecycle.NOVEL

    @property
    def schedule_key(self) -> str:
        return stable_digest(
            "scheduled-case",
            {
                "case_id": self.case_id,
                "case_index": self.case_index,
                "source": self.source.value,
                "derived_seed": self.derived_seed,
                "target_shard": self.target_shard,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_index": self.case_index,
            "source": self.source.value,
            "derived_seed": self.derived_seed,
            "target_shard": self.target_shard,
            "capability_units": list(self.capability_units),
            "utility": self.utility.to_dict(),
            "causal_signature": self.causal_signature,
            "semantic_novelty": (
                self.semantic_novelty.compact_dict()
                if self.semantic_novelty is not None
                else {}
            ),
            "semantic_novelty_preview": (
                self.semantic_novelty_preview.to_dict()
                if self.semantic_novelty_preview is not None
                else {}
            ),
            "family_lifecycle": self.family_lifecycle.value,
            "schedule_key": self.schedule_key,
        }


@dataclass(slots=True)
class CapabilityCell:
    unit: str
    status: CapabilityStatus = CapabilityStatus.UNKNOWN
    evidence_id: str = ""
    minimum_witness: str = ""

    def transition(
        self,
        status: CapabilityStatus,
        *,
        evidence_id: str = "",
        minimum_witness: str = "",
    ) -> None:
        if status is CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE and not evidence_id:
            raise ValueError("unsupported capability cells require evidence")
        if status is CapabilityStatus.WITNESSED and not minimum_witness:
            raise ValueError("witnessed capability cells require a replayable witness")
        self.status = status
        self.evidence_id = str(evidence_id)
        self.minimum_witness = str(minimum_witness)

    def to_dict(self) -> dict[str, str]:
        return {
            "unit": self.unit,
            "status": self.status.value,
            "evidence_id": self.evidence_id,
            "minimum_witness": self.minimum_witness,
        }


@dataclass(frozen=True, slots=True)
class CoordinatorPolicy:
    fresh_min_share: float = 0.60
    feedback_max_share: float = 0.25
    same_case_backend_parallel: bool = False
    recheck_parallelism: int = 2
    coverage_debt_enabled: bool = True
    semantic_novelty_enabled: bool = True
    semantic_novelty_weight: float = 1.0
    semantic_depth_weight: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.fresh_min_share) <= 1.0:
            raise ValueError("fresh_min_share must be in [0, 1]")
        if not 0.0 <= float(self.feedback_max_share) <= 1.0:
            raise ValueError("feedback_max_share must be in [0, 1]")
        if int(self.recheck_parallelism) <= 0:
            raise ValueError("recheck_parallelism must be positive")
        if float(self.semantic_novelty_weight) < 0.0:
            raise ValueError("semantic_novelty_weight must be non-negative")
        if float(self.semantic_depth_weight) < 0.0:
            raise ValueError("semantic_depth_weight must be non-negative")


@dataclass(slots=True)
class SourceReport:
    cases: int = 0
    candidates: int = 0
    independent_roots: set[str] = field(default_factory=set)
    observed_roots: set[str] = field(default_factory=set)
    fresh_rewardable_candidates: int = 0
    excluded_candidate_count: int = 0
    cpu_ms: float = 0.0
    io_bytes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "candidates": self.candidates,
            "independent_root_count": len(self.independent_roots),
            "observed_root_count": len(self.observed_roots),
            "fresh_rewardable_candidates": self.fresh_rewardable_candidates,
            "excluded_candidate_count": self.excluded_candidate_count,
            "cpu_ms": self.cpu_ms,
            "io_bytes": self.io_bytes,
        }


class Coordinator:
    def __init__(
        self,
        capability_units: Iterable[str] = (),
        *,
        policy: CoordinatorPolicy | None = None,
    ) -> None:
        self.policy = policy or CoordinatorPolicy()
        self.cells = {
            str(unit): CapabilityCell(str(unit))
            for unit in sorted({str(unit) for unit in capability_units if str(unit)})
        }
        self.source_reports = {source: SourceReport() for source in CandidateSource}
        self.selected: list[ScheduledCase] = []
        self.known_root_hits: Counter[str] = Counter()
        self.semantic_novelty_ledger = SemanticNoveltyLedger()
        self.lifecycle_counts: Counter[str] = Counter()
        self.evidence_counts: Counter[str] = Counter()

    def coverage_debt(self, capability_units: Iterable[str]) -> float:
        units = [str(unit) for unit in capability_units if str(unit)]
        if not units:
            return 0.0
        debt = 0.0
        for unit in units:
            cell = self.cells.get(unit)
            if cell is None or cell.status is CapabilityStatus.UNKNOWN:
                debt += 1.0
            elif cell.status is CapabilityStatus.BLOCKED:
                debt += 0.25
        return debt / len(units)

    def select_next(self, candidates: Iterable[ScheduledCase]) -> ScheduledCase | None:
        selected = self.rank(candidates)
        if selected is not None:
            self.admit(selected)
        return selected

    def rank(self, candidates: Iterable[ScheduledCase]) -> ScheduledCase | None:
        pool = sorted(candidates, key=lambda item: (item.case_index, item.derived_seed, item.case_id, item.target_shard))
        if not pool:
            return None
        eligible = [item for item in pool if self._source_allowed(item.source, pool)]
        return max(eligible or pool, key=self._selection_key)

    def admit(self, scheduled: ScheduledCase) -> None:
        # A failed execution must remain visible as unknown/blocked evidence.
        # Register capability cells before running the case; otherwise only
        # successful outcomes ever create target-specific cells and coverage
        # can be falsely closed by omission.
        for unit in scheduled.capability_units:
            self.cells.setdefault(str(unit), CapabilityCell(str(unit)))
        self.selected.append(scheduled)
        self.source_reports[scheduled.source].cases += 1
        if scheduled.semantic_novelty is not None:
            self.semantic_novelty_ledger.record(scheduled.semantic_novelty)

    def preview_semantic_novelty(
        self,
        descriptor: SemanticNoveltyDescriptor,
    ) -> SemanticNoveltyPreview:
        return self.semantic_novelty_ledger.preview(descriptor)

    def requires_fresh_source(self) -> bool:
        """Whether the next assigned case must be grammar-generated.

        The decision is made before generation, so a worker never consumes an
        opportunistically assigned feedback seed while the campaign is behind
        on its independent-generation quota.
        """

        selected_count = len(self.selected)
        fresh_cases = self.source_reports[CandidateSource.FRESH_GRAMMAR].cases
        feedback_cases = self.source_reports[CandidateSource.FEEDBACK_MUTATION].cases
        required_fresh = int((selected_count + 1) * self.policy.fresh_min_share + 0.999999)
        allowed_feedback = int((selected_count + 1) * self.policy.feedback_max_share)
        return fresh_cases < required_fresh or feedback_cases >= allowed_feedback

    def record_outcome(
        self,
        scheduled: ScheduledCase,
        *,
        candidate: bool = False,
        causal_signature: str = "",
        causal_signatures: Iterable[str] = (),
        observed_families: Iterable[str] = (),
        cpu_ms: float = 0.0,
        io_bytes: float = 0.0,
    ) -> None:
        report = self.source_reports[scheduled.source]
        report.candidates += int(bool(candidate))
        self.lifecycle_counts[scheduled.family_lifecycle.value] += int(bool(candidate))
        self.evidence_counts[scheduled.source.value] += int(bool(candidate))
        report.cpu_ms += max(0.0, float(cpu_ms))
        report.io_bytes += max(0.0, float(io_bytes))
        signatures = {
            str(signature)
            for signature in (causal_signature, scheduled.causal_signature, *causal_signatures)
            if str(signature)
        }
        for signature in signatures:
            report.observed_roots.add(signature)
        fresh_eligible = (
            scheduled.source
            not in {
                CandidateSource.HISTORICAL_REPLAY,
                CandidateSource.KNOWN_REGRESSION,
                CandidateSource.SEEDED_SENSITIVITY,
            }
            and scheduled.family_lifecycle.fresh_reward_eligible
        )
        if candidate and fresh_eligible:
            report.fresh_rewardable_candidates += 1
            for signature in signatures:
                report.independent_roots.add(signature)
                self.known_root_hits[signature] += 1
        elif candidate:
            report.excluded_candidate_count += 1
        if scheduled.semantic_novelty is not None and scheduled.source is not CandidateSource.SEEDED_SENSITIVITY:
            self.semantic_novelty_ledger.record_roots(
                (
                    *scheduled.semantic_novelty.saturation_keys,
                    *(str(family) for family in observed_families if str(family)),
                ),
                signatures,
            )

    def update_capability(
        self,
        unit: str,
        status: CapabilityStatus,
        *,
        evidence_id: str = "",
        minimum_witness: str = "",
    ) -> None:
        key = str(unit)
        cell = self.cells.get(key)
        if cell is None:
            cell = CapabilityCell(key)
            cell.transition(
                status,
                evidence_id=evidence_id,
                minimum_witness=minimum_witness,
            )
            self.cells[key] = cell
            return
        cell.transition(status, evidence_id=evidence_id, minimum_witness=minimum_witness)

    def snapshot(self, *, omit_recomputable: bool = False) -> dict[str, Any]:
        cells = [self.cells[unit] for unit in sorted(self.cells)]
        status_counts = Counter(cell.status.value for cell in cells)
        semantic_novelty = self.semantic_novelty_ledger.summary()
        generalization_readiness = self.generalization_readiness()
        selected_case_keys = [item.schedule_key for item in self.selected]
        snapshot = {
            "schema_version": "datadiff-coordinator-v1",
            "policy": asdict(self.policy),
            "execution_topology": {
                "same_case_backend_parallel": self.policy.same_case_backend_parallel,
                "recheck_parallelism": self.policy.recheck_parallelism,
                "default": "cross_case_cross_target_shard",
            },
            "source_reports": {
                source.value: report.to_dict()
                for source, report in self.source_reports.items()
            },
            "evidence_accounting": {
                "candidate_counts_by_lane": dict(sorted(self.evidence_counts.items())),
                "candidate_counts_by_lifecycle": dict(sorted(self.lifecycle_counts.items())),
                "fresh_rewardable_candidate_count": sum(
                    report.fresh_rewardable_candidates
                    for report in self.source_reports.values()
                ),
                "excluded_candidate_count": sum(
                    report.excluded_candidate_count
                    for report in self.source_reports.values()
                ),
            },
            "semantic_novelty": semantic_novelty,
            "capability_cells": [cell.to_dict() for cell in cells],
            "coverage": {
                "cell_count": len(self.cells),
                "status_counts": dict(sorted(status_counts.items())),
                "debt_cells": sum(
                    1
                    for cell in self.cells.values()
                    if cell.status in {CapabilityStatus.UNKNOWN, CapabilityStatus.BLOCKED}
                ),
            },
            "coverage_debt": sum(
                1
                for cell in self.cells.values()
                if cell.status in {CapabilityStatus.UNKNOWN, CapabilityStatus.BLOCKED}
            ),
            "generalization_readiness": generalization_readiness,
            "selected_case_keys": selected_case_keys,
        }
        if not omit_recomputable:
            snapshot["coverage_bitmap"] = _coverage_bitmap(cells)
            snapshot["minimum_witnesses"] = [
                {
                    "unit": cell.unit,
                    "status": cell.status.value,
                    "evidence_id": cell.evidence_id,
                    "minimum_witness": cell.minimum_witness,
                }
                for cell in cells
                if cell.evidence_id or cell.minimum_witness
            ]
            return snapshot

        # Minimal run evidence retains the canonical cells and aggregate
        # diagnostics.  The expanded views below are deterministic projections
        # of those cells or of the frozen case stream, and dominate metadata
        # bytes for broad semantic portfolios.
        stratum_counts = dict(semantic_novelty.pop("stratum_counts", {}) or {})
        semantic_novelty.update(
            {
                "stratum_count": len(stratum_counts),
                "stratum_observation_count": sum(
                    int(value or 0) for value in stratum_counts.values()
                ),
                "stratum_counts_digest": stable_digest(
                    "semantic-strata",
                    stratum_counts,
                ),
            }
        )
        unresolved_units = list(
            generalization_readiness.pop("unresolved_units", []) or []
        )
        generalization_readiness["unresolved_units_digest"] = stable_digest(
            "coordinator-unresolved-units",
            unresolved_units,
        )
        snapshot.pop("coverage", None)
        snapshot.pop("coverage_debt", None)
        snapshot.pop("selected_case_keys", None)
        snapshot["selected_case_key_count"] = len(selected_case_keys)
        snapshot["selected_case_keys_digest"] = stable_digest(
            "coordinator-selected-case-keys",
            selected_case_keys,
        )
        snapshot["recomputable_omissions"] = {
            "schema_version": "coordinator-recomputable-omissions-v1",
            "fields": [
                "coverage",
                "coverage_bitmap",
                "coverage_debt",
                "generalization_readiness.unresolved_units",
                "minimum_witnesses",
                "selected_case_keys",
                "semantic_novelty.stratum_counts",
            ],
            "reconstruction_basis": (
                "capability_cells plus the frozen seed manifest, method arm, "
                "and retained run outcomes"
            ),
        }
        return snapshot

    def generalization_readiness(self) -> dict[str, Any]:
        """Fail closed until every declared capability cell has evidence."""

        unknown = sorted(
            cell.unit for cell in self.cells.values() if cell.status is CapabilityStatus.UNKNOWN
        )
        blocked = sorted(
            cell.unit for cell in self.cells.values() if cell.status is CapabilityStatus.BLOCKED
        )
        unsupported = sorted(
            cell.unit
            for cell in self.cells.values()
            if cell.status is CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE
        )
        witnessed = sum(
            cell.status is CapabilityStatus.WITNESSED for cell in self.cells.values()
        )
        return {
            "eligible": not unknown and not blocked,
            "unknown_count": len(unknown),
            "blocked_count": len(blocked),
            "unsupported_with_evidence_count": len(unsupported),
            "witnessed_count": witnessed,
            "unresolved_units": [*unknown, *blocked],
        }

    def _source_allowed(
        self,
        source: CandidateSource,
        pool: list[ScheduledCase],
    ) -> bool:
        selected_count = len(self.selected)
        reports = self.source_reports
        if source is CandidateSource.FRESH_GRAMMAR:
            return True
        fresh_available = any(item.source is CandidateSource.FRESH_GRAMMAR for item in pool)
        required_fresh = int((selected_count + 1) * self.policy.fresh_min_share + 0.999999)
        if fresh_available and reports[CandidateSource.FRESH_GRAMMAR].cases < required_fresh:
            return False
        if source is CandidateSource.FEEDBACK_MUTATION:
            maximum_feedback = int((selected_count + 1) * self.policy.feedback_max_share)
            feedback_available = any(item.source is CandidateSource.FEEDBACK_MUTATION for item in pool)
            if feedback_available and reports[CandidateSource.FEEDBACK_MUTATION].cases >= maximum_feedback:
                return False
        return True

    def _selection_key(
        self,
        candidate: ScheduledCase,
    ) -> tuple[float, float, float, float, float, int, int, str]:
        utility = candidate.utility
        dynamic_score = utility.score + (
            self.coverage_debt(candidate.capability_units)
            if self.policy.coverage_debt_enabled
            else 0.0
        )
        semantic_rarity = 0.0
        semantic_depth = 0.0
        if self.policy.semantic_novelty_enabled and candidate.semantic_novelty is not None:
            # Campaign construction freezes this preview for its whole batch.
            # Hand-built callers can omit it and retain the historical dynamic
            # behaviour used by the coordinator's small unit tests.
            preview = candidate.semantic_novelty_preview or self.semantic_novelty_ledger.preview(
                candidate.semantic_novelty
            )
            semantic_rarity = preview.rarity_score
            semantic_depth = candidate.semantic_novelty.depth_score
            dynamic_score += (
                self.policy.semantic_novelty_weight * semantic_rarity
                + self.policy.semantic_depth_weight * semantic_depth
            )
        saturation = self.known_root_hits[str(candidate.causal_signature)]
        return (
            dynamic_score - saturation,
            semantic_rarity,
            semantic_depth,
            utility.novelty_upper_bound,
            utility.diversity_gain,
            -candidate.case_index,
            -candidate.derived_seed,
            candidate.case_id,
        )


def merge_coordinator_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge shard-local coverage evidence without treating missing work as proof.

    Shards run independently, so a capability witnessed in one block remains
    unknown in all other blocks.  P6 is evaluated over their union.  Opposite
    supported/unsupported conclusions are retained as blocked conflicts rather
    than silently choosing one and producing an unsound closure.
    """

    cells_by_unit: dict[str, list[Mapping[str, Any]]] = {}
    snapshots = [snapshot for snapshot in snapshots if isinstance(snapshot, Mapping)]
    for snapshot in snapshots:
        for cell in snapshot.get("capability_cells", ()):
            if isinstance(cell, Mapping) and str(cell.get("unit", "")):
                cells_by_unit.setdefault(str(cell["unit"]), []).append(cell)

    coordinator = Coordinator(cells_by_unit)
    conflicts: list[dict[str, Any]] = []
    for unit, cells in sorted(cells_by_unit.items()):
        statuses = {
            CapabilityStatus(str(cell.get("status", CapabilityStatus.UNKNOWN.value)))
            for cell in cells
        }
        witnessed = [str(cell.get("minimum_witness", "")) for cell in cells if cell.get("minimum_witness")]
        unsupported = [str(cell.get("evidence_id", "")) for cell in cells if cell.get("evidence_id")]
        if CapabilityStatus.BLOCKED in statuses:
            coordinator.update_capability(unit, CapabilityStatus.BLOCKED)
        elif {
            CapabilityStatus.WITNESSED,
            CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE,
        } <= statuses:
            coordinator.update_capability(unit, CapabilityStatus.BLOCKED)
            conflicts.append(
                {
                    "unit": unit,
                    "statuses": sorted(status.value for status in statuses),
                    "minimum_witnesses": sorted(set(witnessed)),
                    "evidence_ids": sorted(set(unsupported)),
                }
            )
        elif CapabilityStatus.WITNESSED in statuses:
            coordinator.update_capability(
                unit,
                CapabilityStatus.WITNESSED,
                minimum_witness=min(witnessed),
            )
        elif CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE in statuses:
            coordinator.update_capability(
                unit,
                CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE,
                evidence_id=min(unsupported),
            )

    snapshot = coordinator.snapshot()
    readiness = dict(snapshot["generalization_readiness"])
    readiness["conflict_count"] = len(conflicts)
    readiness["conflicting_units"] = [item["unit"] for item in conflicts]
    readiness["eligible"] = bool(readiness["eligible"] and not conflicts)
    snapshot["generalization_readiness"] = readiness
    snapshot["merge"] = {
        "schema_version": "datadiff-coordinator-merge-v1",
        "input_snapshot_count": len(snapshots),
        "conflicts": conflicts,
    }
    return snapshot


def _coverage_bitmap(cells: list[CapabilityCell]) -> dict[str, Any]:
    """Encode four capability states in deterministic two-bit slots."""

    codes = {
        CapabilityStatus.UNKNOWN: 0,
        CapabilityStatus.WITNESSED: 1,
        CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE: 2,
        CapabilityStatus.BLOCKED: 3,
    }
    packed = bytearray((len(cells) + 3) // 4)
    for index, cell in enumerate(cells):
        packed[index // 4] |= codes[cell.status] << ((index % 4) * 2)
    units = [cell.unit for cell in cells]
    return {
        "schema_version": "datadiff-capability-bitmap-v1",
        "encoding": "two-bit-little-endian-hex",
        "status_codes": {
            "0": CapabilityStatus.UNKNOWN.value,
            "1": CapabilityStatus.WITNESSED.value,
            "2": CapabilityStatus.UNSUPPORTED_WITH_EVIDENCE.value,
            "3": CapabilityStatus.BLOCKED.value,
        },
        "unit_count": len(units),
        "unit_digest": stable_digest("capability-units", units),
        "data": packed.hex(),
    }
