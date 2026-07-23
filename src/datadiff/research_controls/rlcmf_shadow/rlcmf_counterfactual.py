from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from datadiff.candidate_burst import candidate_family_evidence
from datadiff.execution_accounting import (
    execution_profile_backend_calls,
    execution_profile_backend_reported_ms,
)
from datadiff.family_novelty import split_family_key
from datadiff.finding_outcomes import candidate_issue_family_keys
from datadiff.process_cpu_accounting import ProcessCPUAccountingLedger
from datadiff.research_controls.rlcmf_shadow.rlcmf_sentinel import (
    RLCMFSentinelCoordinator,
    SentinelAuditPlan,
    SentinelAuditResult,
)


COUNTERFACTUAL_EVENT_METRICS = {
    "any_new_candidate_family",
    "new_candidate_family_count",
    "any_new_candidate_root",
    "new_candidate_root_count",
    "any_new_confirmed_family",
    "new_confirmed_family_count",
    "any_new_confirmed_root",
    "new_confirmed_root_count",
}


class CandidateCommonBatchError(RuntimeError):
    """Raised when a selected candidate audit cannot preserve its contract."""


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_json_copy(payload: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json_bytes(payload).decode("utf-8"))


def _candidate_families(row: Mapping[str, Any]) -> tuple[str, ...]:
    findings = row.get("findings", []) or []
    if not isinstance(findings, list):
        return ()
    return tuple(sorted(candidate_issue_family_keys(findings)))


def _confirmed_candidate_families(row: Mapping[str, Any]) -> tuple[str, ...]:
    evidence = candidate_family_evidence(dict(row))
    return tuple(
        sorted(str(value) for value in evidence["confirmed_candidate_families"])
    )


def _roots(families: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({split_family_key(family)[0] for family in families}))


@dataclass(frozen=True, slots=True)
class CounterfactualCandidateComparison:
    low_status: str
    full_status: str
    low_candidate_families: tuple[str, ...]
    full_candidate_families: tuple[str, ...]
    new_candidate_families: tuple[str, ...]
    low_only_candidate_families: tuple[str, ...]
    low_candidate_roots: tuple[str, ...]
    full_candidate_roots: tuple[str, ...]
    new_candidate_roots: tuple[str, ...]
    low_only_candidate_roots: tuple[str, ...]
    low_confirmed_families: tuple[str, ...]
    full_confirmed_families: tuple[str, ...]
    new_confirmed_families: tuple[str, ...]
    low_only_confirmed_families: tuple[str, ...]
    low_confirmed_roots: tuple[str, ...]
    full_confirmed_roots: tuple[str, ...]
    new_confirmed_roots: tuple[str, ...]
    low_only_confirmed_roots: tuple[str, ...]

    @classmethod
    def compare(
        cls,
        low_row: Mapping[str, Any],
        full_row: Mapping[str, Any],
    ) -> "CounterfactualCandidateComparison":
        low_families = _candidate_families(low_row)
        full_families = _candidate_families(full_row)
        low_roots = _roots(low_families)
        full_roots = _roots(full_families)
        low_confirmed = _confirmed_candidate_families(low_row)
        full_confirmed = _confirmed_candidate_families(full_row)
        low_confirmed_roots = _roots(low_confirmed)
        full_confirmed_roots = _roots(full_confirmed)
        return cls(
            low_status=str(low_row.get("status", "")),
            full_status=str(full_row.get("status", "")),
            low_candidate_families=low_families,
            full_candidate_families=full_families,
            new_candidate_families=tuple(sorted(set(full_families) - set(low_families))),
            low_only_candidate_families=tuple(
                sorted(set(low_families) - set(full_families))
            ),
            low_candidate_roots=low_roots,
            full_candidate_roots=full_roots,
            new_candidate_roots=tuple(sorted(set(full_roots) - set(low_roots))),
            low_only_candidate_roots=tuple(
                sorted(set(low_roots) - set(full_roots))
            ),
            low_confirmed_families=low_confirmed,
            full_confirmed_families=full_confirmed,
            new_confirmed_families=tuple(
                sorted(set(full_confirmed) - set(low_confirmed))
            ),
            low_only_confirmed_families=tuple(
                sorted(set(low_confirmed) - set(full_confirmed))
            ),
            low_confirmed_roots=low_confirmed_roots,
            full_confirmed_roots=full_confirmed_roots,
            new_confirmed_roots=tuple(
                sorted(set(full_confirmed_roots) - set(low_confirmed_roots))
            ),
            low_only_confirmed_roots=tuple(
                sorted(set(low_confirmed_roots) - set(full_confirmed_roots))
            ),
        )

    def event_value(self, metric: str) -> float:
        metric = str(metric)
        if metric not in COUNTERFACTUAL_EVENT_METRICS:
            raise ValueError(f"unsupported counterfactual event metric: {metric}")
        values = {
            "new_candidate_family_count": len(self.new_candidate_families),
            "new_candidate_root_count": len(self.new_candidate_roots),
            "new_confirmed_family_count": len(self.new_confirmed_families),
            "new_confirmed_root_count": len(self.new_confirmed_roots),
        }
        if metric.startswith("any_"):
            count_metric = metric.removeprefix("any_") + "_count"
            return float(values[count_metric] > 0)
        return float(values[metric])

    def full_event_value(self, metric: str) -> float:
        collections = {
            "candidate_family": self.full_candidate_families,
            "candidate_root": self.full_candidate_roots,
            "confirmed_family": self.full_confirmed_families,
            "confirmed_root": self.full_confirmed_roots,
        }
        return self._scope_value(metric, collections)

    def potential_misclassification_value(self, metric: str) -> float:
        collections = {
            "candidate_family": self.low_only_candidate_families,
            "candidate_root": self.low_only_candidate_roots,
            "confirmed_family": self.low_only_confirmed_families,
            "confirmed_root": self.low_only_confirmed_roots,
        }
        return self._scope_value(metric, collections)

    @staticmethod
    def _scope_value(
        metric: str,
        collections: Mapping[str, tuple[str, ...]],
    ) -> float:
        normalized = str(metric)
        is_any = normalized.startswith("any_")
        scope = normalized.removeprefix("any_").removeprefix("new_")
        scope = scope.removesuffix("_count")
        if scope not in collections:
            raise ValueError(f"unsupported counterfactual event metric: {metric}")
        count = len(collections[scope])
        return float(count > 0) if is_any else float(count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-counterfactual-candidate-comparison-v1",
            "estimand_scope": "candidate_events_not_confirmed_real_bugs",
            "low_status": self.low_status,
            "full_status": self.full_status,
            "low_candidate_families": list(self.low_candidate_families),
            "full_candidate_families": list(self.full_candidate_families),
            "new_candidate_families": list(self.new_candidate_families),
            "low_only_candidate_families": list(self.low_only_candidate_families),
            "low_candidate_roots": list(self.low_candidate_roots),
            "full_candidate_roots": list(self.full_candidate_roots),
            "new_candidate_roots": list(self.new_candidate_roots),
            "low_only_candidate_roots": list(self.low_only_candidate_roots),
            "low_confirmed_families": list(self.low_confirmed_families),
            "full_confirmed_families": list(self.full_confirmed_families),
            "new_confirmed_families": list(self.new_confirmed_families),
            "low_only_confirmed_families": list(
                self.low_only_confirmed_families
            ),
            "low_confirmed_roots": list(self.low_confirmed_roots),
            "full_confirmed_roots": list(self.full_confirmed_roots),
            "new_confirmed_roots": list(self.new_confirmed_roots),
            "low_only_confirmed_roots": list(self.low_only_confirmed_roots),
        }


@dataclass(frozen=True, slots=True)
class CandidateCommonBatch:
    """One ordered, digest-addressed candidate batch from a frozen frontier."""

    frontier_id: str
    items: tuple[Any, ...] = field(repr=False, compare=False)
    canonical_payloads: tuple[Mapping[str, Any], ...] = field(repr=False)
    generation_trace: Mapping[str, Any] = field(default_factory=dict, repr=False)
    candidate_sha256s: tuple[str, ...] = field(init=False)
    batch_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        frontier_id = str(self.frontier_id or "").strip()
        if not frontier_id:
            raise ValueError("candidate batch frontier_id cannot be empty")
        if not self.items:
            raise ValueError("candidate common batch cannot be empty")
        if len(self.items) != len(self.canonical_payloads):
            raise ValueError("candidate items and canonical payloads must align")
        payloads = tuple(_canonical_json_copy(payload) for payload in self.canonical_payloads)
        trace = _canonical_json_copy(self.generation_trace)
        candidate_sha256s = tuple(
            hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
            for payload in payloads
        )
        batch_material = {
            "schema_version": "rlcmf-candidate-common-batch-id-v1",
            "frontier_id": frontier_id,
            "candidate_sha256s": list(candidate_sha256s),
            "generation_trace": trace,
        }
        object.__setattr__(self, "frontier_id", frontier_id)
        object.__setattr__(self, "canonical_payloads", payloads)
        object.__setattr__(self, "generation_trace", trace)
        object.__setattr__(self, "candidate_sha256s", candidate_sha256s)
        object.__setattr__(
            self,
            "batch_sha256",
            hashlib.sha256(_canonical_json_bytes(batch_material)).hexdigest(),
        )

    @classmethod
    def from_items(
        cls,
        *,
        frontier_id: str,
        items: Sequence[Any],
        payload_fn: Callable[[Any], Mapping[str, Any]],
        generation_trace: Mapping[str, Any] | None = None,
    ) -> "CandidateCommonBatch":
        normalized_items = tuple(items)
        return cls(
            frontier_id=frontier_id,
            items=normalized_items,
            canonical_payloads=tuple(payload_fn(item) for item in normalized_items),
            generation_trace=dict(generation_trace or {}),
        )

    def candidate_id(self, index: int) -> str:
        normalized_index = int(index)
        if normalized_index < 0 or normalized_index >= len(self.items):
            raise IndexError("candidate index is outside the common batch")
        material = {
            "schema_version": "rlcmf-candidate-opportunity-id-v1",
            "batch_sha256": self.batch_sha256,
            "candidate_index": normalized_index,
            "candidate_sha256": self.candidate_sha256s[normalized_index],
        }
        return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-candidate-common-batch-v1",
            "frontier_id": self.frontier_id,
            "candidate_count": len(self.items),
            "candidate_sha256s": list(self.candidate_sha256s),
            "candidate_ids": [self.candidate_id(index) for index in range(len(self.items))],
            "batch_sha256": self.batch_sha256,
            "generation_trace": dict(self.generation_trace),
        }


@dataclass(frozen=True, slots=True)
class CandidatePolicySelection:
    """A policy choice made only from a supplied frozen snapshot and batch view."""

    selected_index: int
    policy_id: str
    snapshot_sha256: str
    score: float = 0.0
    trace: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.selected_index) < 0:
            raise ValueError("candidate selected_index must be non-negative")
        if not str(self.policy_id or "").strip():
            raise ValueError("candidate policy_id cannot be empty")
        normalized_digest = str(self.snapshot_sha256 or "").strip().lower()
        if len(normalized_digest) != 64 or any(
            char not in "0123456789abcdef" for char in normalized_digest
        ):
            raise ValueError("candidate snapshot_sha256 must be a SHA-256 digest")
        if not math.isfinite(float(self.score)):
            raise ValueError("candidate selection score must be finite")
        object.__setattr__(self, "policy_id", str(self.policy_id).strip())
        object.__setattr__(self, "snapshot_sha256", normalized_digest)
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "trace", _canonical_json_copy(self.trace))


@dataclass(frozen=True, slots=True)
class CandidateSelectionRecord:
    arm: str
    policy_id: str
    snapshot_sha256: str
    selected_index: int
    candidate_id: str
    candidate_sha256: str
    score: float
    trace: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-candidate-selection-record-v1",
            "arm": self.arm,
            "policy_id": self.policy_id,
            "snapshot_sha256": self.snapshot_sha256,
            "selected_index": self.selected_index,
            "candidate_id": self.candidate_id,
            "candidate_sha256": self.candidate_sha256,
            "score": self.score,
            "trace": dict(self.trace),
        }


@dataclass(slots=True)
class CandidateCommonBatchAuditResult:
    opportunity: dict[str, Any]
    plan: SentinelAuditPlan
    low_batch: CandidateCommonBatch
    common_batch: CandidateCommonBatch | None
    low_selection: CandidateSelectionRecord
    full_selection: CandidateSelectionRecord | None
    low_row: dict[str, Any]
    full_shadow_row: dict[str, Any] | None
    comparison: CounterfactualCandidateComparison | None
    sentinel_result: SentinelAuditResult
    full_shadow_reused_from_low: bool

    def summary(self) -> dict[str, Any]:
        low_calls = execution_profile_backend_calls(
            self.low_row.get("execution_profile", {})
        )
        shadow_calls = (
            0
            if self.full_shadow_row is None or self.full_shadow_reused_from_low
            else execution_profile_backend_calls(
                self.full_shadow_row.get("execution_profile", {})
            )
        )
        low_backend_ms = execution_profile_backend_reported_ms(
            self.low_row.get("execution_profile", {})
        )
        shadow_backend_ms = (
            0.0
            if self.full_shadow_row is None or self.full_shadow_reused_from_low
            else execution_profile_backend_reported_ms(
                self.full_shadow_row.get("execution_profile", {})
            )
        )
        low_wall_ms = float(self.low_row.get("duration_ms", 0.0) or 0.0)
        shadow_wall_ms = (
            0.0
            if self.full_shadow_row is None or self.full_shadow_reused_from_low
            else float(self.full_shadow_row.get("duration_ms", 0.0) or 0.0)
        )
        return {
            "schema_version": "rlcmf-candidate-common-batch-summary-v1",
            "opportunity_id": self.opportunity["opportunity_id"],
            "selected": self.plan.selected,
            "propensity_exact": self.plan.decision.to_dict()["propensity_exact"],
            "created_debt_id": self.plan.created_debt_id,
            "settled_debt_id": self.sentinel_result.settled_debt_id,
            "common_batch_sha256": (
                self.common_batch.batch_sha256 if self.common_batch is not None else None
            ),
            "low_candidate_id": self.low_selection.candidate_id,
            "full_candidate_id": (
                self.full_selection.candidate_id
                if self.full_selection is not None
                else None
            ),
            "full_shadow_executed": bool(
                self.full_shadow_row is not None and not self.full_shadow_reused_from_low
            ),
            "full_shadow_reused_from_low": self.full_shadow_reused_from_low,
            "low_backend_calls": low_calls,
            "full_shadow_backend_calls": shadow_calls,
            "combined_backend_calls": low_calls + shadow_calls,
            "low_backend_reported_ms": low_backend_ms,
            "full_shadow_backend_reported_ms": shadow_backend_ms,
            "combined_backend_reported_ms": low_backend_ms + shadow_backend_ms,
            "low_wall_ms": low_wall_ms,
            "full_shadow_wall_ms": shadow_wall_ms,
            "combined_wall_ms": low_wall_ms + shadow_wall_ms,
            "comparison": (
                self.comparison.to_dict() if self.comparison is not None else None
            ),
            "sentinel_result": self.sentinel_result.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-candidate-common-batch-result-v1",
            "opportunity": dict(self.opportunity),
            "plan": self.plan.to_dict(),
            "low_batch": self.low_batch.to_dict(),
            "common_batch": (
                self.common_batch.to_dict() if self.common_batch is not None else None
            ),
            "low_selection": self.low_selection.to_dict(),
            "full_selection": (
                self.full_selection.to_dict()
                if self.full_selection is not None
                else None
            ),
            "low_row": self.low_row,
            "full_shadow_row": self.full_shadow_row,
            "full_shadow_reused_from_low": self.full_shadow_reused_from_low,
            "summary": self.summary(),
        }


class RLCMFCandidateCommonBatchExecutor:
    """Audit candidate-pool omissions on one outcome-blind common batch."""

    def __init__(
        self,
        *,
        coordinator: RLCMFSentinelCoordinator,
        generate_low_batch_fn: Callable[[int], CandidateCommonBatch],
        generate_full_batch_fn: Callable[[int], CandidateCommonBatch],
        select_low_fn: Callable[
            [tuple[Any, ...], Mapping[str, Any]], CandidatePolicySelection
        ],
        select_full_fn: Callable[
            [tuple[Any, ...], Mapping[str, Any]], CandidatePolicySelection
        ],
        run_candidate_fn: Callable[..., dict[str, Any]],
        prediction_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float]
        | None = None,
    ) -> None:
        if "candidate" not in coordinator.event_accumulators:
            raise ValueError("candidate common-batch executor requires candidate accumulator")
        metric = coordinator.event_accumulators["candidate"].metric
        if metric not in COUNTERFACTUAL_EVENT_METRICS:
            raise ValueError("unknown candidate counterfactual event metric")
        self.coordinator = coordinator
        self.event_metric = metric
        self.generate_low_batch_fn = generate_low_batch_fn
        self.generate_full_batch_fn = generate_full_batch_fn
        self.select_low_fn = select_low_fn
        self.select_full_fn = select_full_fn
        self.run_candidate_fn = run_candidate_fn
        self.prediction_fn = prediction_fn or (lambda _opportunity, _row: 0.0)

    @staticmethod
    def _selection_record(
        *,
        arm: str,
        batch: CandidateCommonBatch,
        selection: CandidatePolicySelection,
        snapshot_sha256: str,
        visible_count: int,
    ) -> CandidateSelectionRecord:
        if selection.snapshot_sha256 != snapshot_sha256:
            raise CandidateCommonBatchError(
                f"{arm} selector did not bind to the frozen policy snapshot"
            )
        index = int(selection.selected_index)
        if index < 0 or index >= int(visible_count):
            raise CandidateCommonBatchError(
                f"{arm} selector chose a candidate outside its registered batch view"
            )
        return CandidateSelectionRecord(
            arm=arm,
            policy_id=selection.policy_id,
            snapshot_sha256=snapshot_sha256,
            selected_index=index,
            candidate_id=batch.candidate_id(index),
            candidate_sha256=batch.candidate_sha256s[index],
            score=selection.score,
            trace=selection.trace,
        )

    def _opportunity(
        self,
        *,
        plan: SentinelAuditPlan,
        frontier_id: str,
        low_pool_size: int,
        full_pool_size: int,
        snapshot_sha256: str,
    ) -> dict[str, Any]:
        material = {
            "schema_version": "rlcmf-candidate-common-batch-opportunity-id-v1",
            "manifest_sha256": self.coordinator.manifest_sha256,
            "audit_decision_id": plan.decision.decision_id,
            "frontier_id": frontier_id,
            "low_pool_size": low_pool_size,
            "full_pool_size": full_pool_size,
            "policy_snapshot_sha256": snapshot_sha256,
        }
        return {
            "schema_version": "rlcmf-candidate-common-batch-opportunity-v1",
            "opportunity_id": hashlib.sha256(
                _canonical_json_bytes(material)
            ).hexdigest(),
            "axis": "candidate",
            "frontier_id": frontier_id,
            "low_pool_size": low_pool_size,
            "full_pool_size": full_pool_size,
            "estimated_omitted_candidates": full_pool_size - low_pool_size,
            "policy_snapshot_sha256": snapshot_sha256,
            "audit_decision_id": plan.decision.decision_id,
            "audit_selected": plan.selected,
            "propensity_exact": plan.decision.to_dict()["propensity_exact"],
            "requested_propensity_exact": plan.decision.to_dict()[
                "requested_propensity_exact"
            ],
            "created_debt_id": plan.created_debt_id,
            "forced_debt_id": plan.forced_debt_id,
            "pre_outcome_decision": True,
        }

    def execute(
        self,
        *,
        plan: SentinelAuditPlan,
        frontier_id: str,
        low_pool_size: int,
        full_pool_size: int,
        policy_snapshot: Mapping[str, Any],
    ) -> CandidateCommonBatchAuditResult:
        if plan.axis != "candidate":
            raise ValueError("candidate common-batch executor requires candidate plan")
        registered = self.coordinator.plans.get(plan.plan_id)
        if registered != plan:
            raise ValueError("candidate plan is not registered by this coordinator")
        normalized_frontier = str(frontier_id or "").strip()
        if not normalized_frontier or plan.item_key != normalized_frontier:
            raise ValueError("candidate plan item_key must equal the frozen frontier_id")
        low_size = int(low_pool_size)
        full_size = int(full_pool_size)
        if low_size <= 0 or full_size <= low_size:
            raise ValueError("candidate audit requires 0 < low_pool_size < full_pool_size")
        snapshot = _canonical_json_copy(policy_snapshot)
        snapshot_sha256 = hashlib.sha256(_canonical_json_bytes(snapshot)).hexdigest()
        opportunity = self._opportunity(
            plan=plan,
            frontier_id=normalized_frontier,
            low_pool_size=low_size,
            full_pool_size=full_size,
            snapshot_sha256=snapshot_sha256,
        )
        if float(plan.decision.propensity) <= 0:
            raise CandidateCommonBatchError("candidate opportunity lacks nonzero propensity")

        if plan.selected:
            try:
                common_batch = self.generate_full_batch_fn(full_size)
            except Exception as exc:  # noqa: BLE001 - preserve fail-closed boundary
                raise CandidateCommonBatchError(
                    "selected candidate audit could not generate its common full batch"
                ) from exc
            if not isinstance(common_batch, CandidateCommonBatch):
                raise CandidateCommonBatchError(
                    "selected candidate audit generator returned an invalid batch"
                )
            if (
                common_batch.frontier_id != normalized_frontier
                or len(common_batch.items) != full_size
            ):
                raise CandidateCommonBatchError(
                    "selected candidate audit could not reconstruct the frozen full frontier"
                )
            low_view = common_batch.items[:low_size]
            full_view = common_batch.items
            low_selection = self._selection_record(
                arm="candidate_low",
                batch=common_batch,
                selection=self.select_low_fn(
                    low_view,
                    _canonical_json_copy(snapshot),
                ),
                snapshot_sha256=snapshot_sha256,
                visible_count=low_size,
            )
            full_selection = self._selection_record(
                arm="candidate_full_shadow",
                batch=common_batch,
                selection=self.select_full_fn(
                    full_view,
                    _canonical_json_copy(snapshot),
                ),
                snapshot_sha256=snapshot_sha256,
                visible_count=full_size,
            )
            low_row = self.run_candidate_fn(
                arm="candidate_low",
                candidate=common_batch.items[low_selection.selected_index],
                full_confirmation=True,
                fresh_backend_instances=True,
            )
            if not isinstance(low_row, dict):
                raise CandidateCommonBatchError("candidate low execution returned no row")
            reused = low_selection.candidate_id == full_selection.candidate_id
            if reused:
                full_shadow_row = low_row
            else:
                full_shadow_row = self.run_candidate_fn(
                    arm="candidate_full_shadow",
                    candidate=common_batch.items[full_selection.selected_index],
                    full_confirmation=True,
                    fresh_backend_instances=True,
                )
                if not isinstance(full_shadow_row, dict):
                    raise CandidateCommonBatchError(
                        "candidate full-shadow execution returned no row"
                    )
            comparison = CounterfactualCandidateComparison.compare(
                low_row,
                full_shadow_row,
            )
            prediction = float(self.prediction_fn(opportunity, low_row))
            safety_kwargs: dict[str, float | None] = {}
            if "candidate" in self.coordinator.safety_trackers:
                safety_kwargs = {
                    "full_event": comparison.full_event_value(self.event_metric),
                    "potential_misclassification": (
                        comparison.potential_misclassification_value(self.event_metric)
                    ),
                }
            sentinel_result = self.coordinator.finalize(
                plan,
                outcome=comparison.event_value(self.event_metric),
                prediction=prediction,
                **safety_kwargs,
            )
            return CandidateCommonBatchAuditResult(
                opportunity=opportunity,
                plan=plan,
                low_batch=common_batch,
                common_batch=common_batch,
                low_selection=low_selection,
                full_selection=full_selection,
                low_row=low_row,
                full_shadow_row=full_shadow_row,
                comparison=comparison,
                sentinel_result=sentinel_result,
                full_shadow_reused_from_low=reused,
            )

        low_batch = self.generate_low_batch_fn(low_size)
        if not isinstance(low_batch, CandidateCommonBatch):
            raise CandidateCommonBatchError("candidate low generator returned an invalid batch")
        if low_batch.frontier_id != normalized_frontier or len(low_batch.items) != low_size:
            raise CandidateCommonBatchError(
                "candidate low generator did not preserve the frozen frontier"
            )
        low_selection = self._selection_record(
            arm="candidate_low",
            batch=low_batch,
            selection=self.select_low_fn(
                low_batch.items,
                _canonical_json_copy(snapshot),
            ),
            snapshot_sha256=snapshot_sha256,
            visible_count=low_size,
        )
        low_row = self.run_candidate_fn(
            arm="candidate_low",
            candidate=low_batch.items[low_selection.selected_index],
            full_confirmation=False,
            fresh_backend_instances=False,
        )
        if not isinstance(low_row, dict):
            raise CandidateCommonBatchError("candidate low execution returned no row")
        prediction = float(self.prediction_fn(opportunity, low_row))
        sentinel_result = self.coordinator.finalize(
            plan,
            outcome=None,
            prediction=prediction,
        )
        return CandidateCommonBatchAuditResult(
            opportunity=opportunity,
            plan=plan,
            low_batch=low_batch,
            common_batch=None,
            low_selection=low_selection,
            full_selection=None,
            low_row=low_row,
            full_shadow_row=None,
            comparison=None,
            sentinel_result=sentinel_result,
            full_shadow_reused_from_low=False,
        )


@dataclass(slots=True)
class RLCMFCoupledAuditResult:
    coupling_id: str
    low_batch: CandidateCommonBatch
    common_batch: CandidateCommonBatch | None
    low_selection: CandidateSelectionRecord
    full_selection: CandidateSelectionRecord | None
    rows: dict[str, dict[str, Any]]
    comparisons: dict[str, CounterfactualCandidateComparison]
    sentinel_results: dict[str, SentinelAuditResult]
    reuse_trace: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        executed_nodes = self.reuse_trace["executed_nodes"]
        backend_calls = sum(
            execution_profile_backend_calls(
                self.rows[node].get("execution_profile", {})
            )
            for node in executed_nodes
        )
        backend_reported_ms = sum(
            execution_profile_backend_reported_ms(
                self.rows[node].get("execution_profile", {})
            )
            for node in executed_nodes
        )
        wall_ms = sum(
            float(self.rows[node].get("duration_ms", 0.0) or 0.0)
            for node in executed_nodes
        )
        process_costs = [
            self.rows[node].get("rlcmf_process_cost")
            for node in executed_nodes
            if isinstance(self.rows[node].get("rlcmf_process_cost"), Mapping)
        ]
        process_cpu_ms = sum(
            float(cost.get("total_process_cpu_ms", 0.0) or 0.0)
            for cost in process_costs
        )
        return {
            "schema_version": "rlcmf-coupled-audit-summary-v1",
            "coupling_id": self.coupling_id,
            "common_batch_sha256": (
                self.common_batch.batch_sha256 if self.common_batch is not None else None
            ),
            "executed_node_count": len(executed_nodes),
            "derived_node_count": len(self.reuse_trace["derived_nodes"]),
            "naive_selected_audit_execution_count": self.reuse_trace[
                "naive_selected_audit_execution_count"
            ],
            "actual_selected_audit_execution_count": self.reuse_trace[
                "actual_selected_audit_execution_count"
            ],
            "duplicate_audit_executions_avoided": self.reuse_trace[
                "duplicate_audit_executions_avoided"
            ],
            "backend_calls": backend_calls,
            "backend_reported_ms": backend_reported_ms,
            "wall_ms": wall_ms,
            "process_cpu_available": len(process_costs) == len(executed_nodes),
            "process_cpu_ms": (
                process_cpu_ms if len(process_costs) == len(executed_nodes) else None
            ),
            "comparisons": {
                axis: comparison.to_dict()
                for axis, comparison in sorted(self.comparisons.items())
            },
            "sentinel_results": {
                axis: result.to_dict()
                for axis, result in sorted(self.sentinel_results.items())
            },
            "reuse_trace": self.reuse_trace,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-coupled-audit-result-v1",
            "coupling_id": self.coupling_id,
            "low_batch": self.low_batch.to_dict(),
            "common_batch": (
                self.common_batch.to_dict() if self.common_batch is not None else None
            ),
            "low_selection": self.low_selection.to_dict(),
            "full_selection": (
                self.full_selection.to_dict()
                if self.full_selection is not None
                else None
            ),
            "rows": self.rows,
            "summary": self.summary(),
        }


class RLCMFCoupledCounterfactualExecutor:
    """Execute a coupled four-axis audit with A/B/C full-path reuse."""

    REQUIRED_AXES = ("candidate", "backend", "relation", "joint")

    def __init__(
        self,
        *,
        coordinator: RLCMFSentinelCoordinator,
        generate_low_batch_fn: Callable[[int], CandidateCommonBatch],
        generate_full_batch_fn: Callable[[int], CandidateCommonBatch],
        select_low_fn: Callable[
            [tuple[Any, ...], Mapping[str, Any]], CandidatePolicySelection
        ],
        select_full_fn: Callable[
            [tuple[Any, ...], Mapping[str, Any]], CandidatePolicySelection
        ],
        run_pipeline_fn: Callable[..., dict[str, Any]],
        factorial_deriver: Callable[..., dict[str, Any]] | None = None,
        prediction_fn: Callable[[str, Mapping[str, Any]], float] | None = None,
        process_cpu_ledger: ProcessCPUAccountingLedger | None = None,
        require_process_cpu_accounting: bool = False,
    ) -> None:
        missing = set(self.REQUIRED_AXES) - set(coordinator.event_accumulators)
        if missing:
            raise ValueError(f"coupled executor lacks event accumulators: {sorted(missing)}")
        metrics = {
            axis: coordinator.event_accumulators[axis].metric
            for axis in self.REQUIRED_AXES
        }
        if any(metric not in COUNTERFACTUAL_EVENT_METRICS for metric in metrics.values()):
            raise ValueError("unknown coupled counterfactual event metric")
        self.coordinator = coordinator
        self.event_metrics = metrics
        self.generate_low_batch_fn = generate_low_batch_fn
        self.generate_full_batch_fn = generate_full_batch_fn
        self.select_low_fn = select_low_fn
        self.select_full_fn = select_full_fn
        self.run_pipeline_fn = run_pipeline_fn
        self.factorial_deriver = factorial_deriver
        self.prediction_fn = prediction_fn or (lambda _axis, _row: 0.0)
        self.process_cpu_ledger = process_cpu_ledger
        self.require_process_cpu_accounting = bool(require_process_cpu_accounting)
        if self.require_process_cpu_accounting and process_cpu_ledger is None:
            raise ValueError(
                "required coupled process CPU accounting ledger is unavailable"
            )

    def _finalize_axis(
        self,
        *,
        axis: str,
        plan: SentinelAuditPlan,
        low_row: Mapping[str, Any],
        full_row: Mapping[str, Any] | None,
    ) -> tuple[CounterfactualCandidateComparison | None, SentinelAuditResult]:
        prediction = float(self.prediction_fn(axis, low_row))
        if not plan.selected:
            return None, self.coordinator.finalize(
                plan,
                outcome=None,
                prediction=prediction,
            )
        if full_row is None:
            raise CandidateCommonBatchError(
                f"selected coupled axis lacks a full outcome: {axis}"
            )
        comparison = CounterfactualCandidateComparison.compare(low_row, full_row)
        safety_kwargs: dict[str, float | None] = {}
        if axis in self.coordinator.safety_trackers:
            safety_kwargs = {
                "full_event": comparison.full_event_value(self.event_metrics[axis]),
                "potential_misclassification": (
                    comparison.potential_misclassification_value(
                        self.event_metrics[axis]
                    )
                ),
            }
        result = self.coordinator.finalize(
            plan,
            outcome=comparison.event_value(self.event_metrics[axis]),
            prediction=prediction,
            **safety_kwargs,
        )
        return comparison, result

    def execute(
        self,
        *,
        plans: Mapping[str, SentinelAuditPlan],
        frontier_id: str,
        low_pool_size: int,
        full_pool_size: int,
        policy_snapshot: Mapping[str, Any],
    ) -> RLCMFCoupledAuditResult:
        normalized_plans = dict(plans)
        if set(normalized_plans) != set(self.REQUIRED_AXES):
            raise ValueError("coupled executor requires all four registered axes")
        coupling_ids = {plan.coupling_id for plan in normalized_plans.values()}
        if len(coupling_ids) != 1 or not next(iter(coupling_ids)):
            raise ValueError("coupled executor plans must share one coupling_id")
        coupling_id = next(iter(coupling_ids))
        decision_set = self.coordinator.coupled_decision_sets.get(coupling_id)
        if decision_set is None or not self.coordinator.auditor.verify_coupled(
            decision_set
        ):
            raise ValueError("coupled executor lacks a verified coupled decision set")
        if any(
            self.coordinator.plans.get(plan.plan_id) != plan
            for plan in normalized_plans.values()
        ):
            raise ValueError("coupled executor received an unregistered plan")
        if normalized_plans["joint"].selected and not all(
            normalized_plans[axis].selected
            for axis in ("candidate", "backend", "relation")
        ):
            raise CandidateCommonBatchError(
                "joint selection is not nested inside all coupled component audits"
            )
        normalized_frontier = str(frontier_id or "").strip()
        if normalized_plans["candidate"].item_key != normalized_frontier:
            raise ValueError("candidate plan item_key must equal the frozen frontier_id")
        low_size = int(low_pool_size)
        full_size = int(full_pool_size)
        if low_size <= 0 or full_size <= low_size:
            raise ValueError("coupled candidate audit requires a reduced/full pool gap")
        snapshot = _canonical_json_copy(policy_snapshot)
        snapshot_sha256 = hashlib.sha256(_canonical_json_bytes(snapshot)).hexdigest()

        candidate_selected = normalized_plans["candidate"].selected
        if candidate_selected:
            batch = self.generate_full_batch_fn(full_size)
            if (
                not isinstance(batch, CandidateCommonBatch)
                or batch.frontier_id != normalized_frontier
                or len(batch.items) != full_size
            ):
                raise CandidateCommonBatchError(
                    "coupled audit could not reconstruct the candidate full frontier"
                )
            common_batch: CandidateCommonBatch | None = batch
            visible_low = low_size
        else:
            batch = self.generate_low_batch_fn(low_size)
            if (
                not isinstance(batch, CandidateCommonBatch)
                or batch.frontier_id != normalized_frontier
                or len(batch.items) != low_size
            ):
                raise CandidateCommonBatchError(
                    "coupled audit could not preserve the candidate low frontier"
                )
            common_batch = None
            visible_low = low_size

        low_selection = RLCMFCandidateCommonBatchExecutor._selection_record(
            arm="adaptive_low",
            batch=batch,
            selection=self.select_low_fn(
                batch.items[:visible_low],
                _canonical_json_copy(snapshot),
            ),
            snapshot_sha256=snapshot_sha256,
            visible_count=visible_low,
        )
        full_selection = None
        if candidate_selected:
            full_selection = RLCMFCandidateCommonBatchExecutor._selection_record(
                arm="full_static_shadow",
                batch=batch,
                selection=self.select_full_fn(
                    batch.items,
                    _canonical_json_copy(snapshot),
                ),
                snapshot_sha256=snapshot_sha256,
                visible_count=full_size,
            )

        low_candidate = batch.items[low_selection.selected_index]
        rows: dict[str, dict[str, Any]] = {}
        executed_nodes: list[str] = []
        derived_nodes: list[str] = []
        node_sources: dict[str, str] = {}

        def execute_node(
            node: str,
            *,
            candidate: Any,
            backend_fidelity: str,
            relation_fidelity: str,
            full_confirmation: bool,
            fresh_backend_instances: bool,
        ) -> dict[str, Any]:
            scope_id = f"{coupling_id}:{node}"
            if self.process_cpu_ledger is not None:
                self.process_cpu_ledger.start_scope(
                    scope_id,
                    metadata={
                        "coupling_id": coupling_id,
                        "node": node,
                        "backend_fidelity": backend_fidelity,
                        "relation_fidelity": relation_fidelity,
                    },
                )
            try:
                row = self.run_pipeline_fn(
                    arm=node,
                    candidate=candidate,
                    backend_fidelity=backend_fidelity,
                    relation_fidelity=relation_fidelity,
                    full_confirmation=full_confirmation,
                    fresh_backend_instances=fresh_backend_instances,
                    process_cpu_ledger=self.process_cpu_ledger,
                    cost_scope_id=scope_id,
                )
            except Exception:
                if self.process_cpu_ledger is not None:
                    self.process_cpu_ledger.finish_scope(status="error")
                raise
            if not isinstance(row, dict):
                if self.process_cpu_ledger is not None:
                    self.process_cpu_ledger.finish_scope(status="invalid_row")
                raise CandidateCommonBatchError(
                    f"coupled pipeline node returned no row: {node}"
                )
            if self.process_cpu_ledger is not None:
                cost_record = self.process_cpu_ledger.finish_scope(
                    status=str(row.get("status", "ok") or "ok"),
                    backend_reported_ms=execution_profile_backend_reported_ms(
                        row.get("execution_profile", {})
                    ),
                    backend_calls=execution_profile_backend_calls(
                        row.get("execution_profile", {})
                    ),
                )
                row = dict(row)
                row["rlcmf_process_cost"] = cost_record.to_dict()
            rows[node] = row
            executed_nodes.append(node)
            node_sources[node] = "executed"
            return row

        adaptive_low = execute_node(
            "adaptive_low",
            candidate=low_candidate,
            backend_fidelity="low",
            relation_fidelity="low",
            full_confirmation=False,
            fresh_backend_instances=False,
        )

        candidate_low_full = None
        full_static = None
        if candidate_selected:
            candidate_low_full = execute_node(
                "candidate_low_full_pipeline",
                candidate=low_candidate,
                backend_fidelity="full",
                relation_fidelity="full",
                full_confirmation=True,
                fresh_backend_instances=True,
            )
            assert full_selection is not None
            if full_selection.candidate_id == low_selection.candidate_id:
                full_static = candidate_low_full
                rows["full_static_shadow"] = full_static
                node_sources["full_static_shadow"] = (
                    "reused_from:candidate_low_full_pipeline"
                )
            else:
                full_static = execute_node(
                    "full_static_shadow",
                    candidate=batch.items[full_selection.selected_index],
                    backend_fidelity="full",
                    relation_fidelity="full",
                    full_confirmation=True,
                    fresh_backend_instances=True,
                )

        component_rows: dict[str, dict[str, Any] | None] = {
            "backend": None,
            "relation": None,
        }
        for axis, backend_fidelity, relation_fidelity in (
            ("backend", "full", "low"),
            ("relation", "low", "full"),
        ):
            if not normalized_plans[axis].selected:
                continue
            node = f"{axis}_full"
            if candidate_low_full is not None and self.factorial_deriver is not None:
                row = self.factorial_deriver(
                    axis=axis,
                    full_pipeline_row=candidate_low_full,
                    candidate=low_candidate,
                )
                if not isinstance(row, dict):
                    raise CandidateCommonBatchError(
                        f"coupled factorial derivation returned no row: {axis}"
                    )
                rows[node] = row
                derived_nodes.append(node)
                node_sources[node] = "derived_from:candidate_low_full_pipeline"
                component_rows[axis] = row
            else:
                component_rows[axis] = execute_node(
                    node,
                    candidate=low_candidate,
                    backend_fidelity=backend_fidelity,
                    relation_fidelity=relation_fidelity,
                    full_confirmation=True,
                    fresh_backend_instances=True,
                )

        comparisons: dict[str, CounterfactualCandidateComparison] = {}
        sentinel_results: dict[str, SentinelAuditResult] = {}
        full_rows = {
            "candidate": full_static,
            "backend": component_rows["backend"],
            "relation": component_rows["relation"],
            "joint": full_static,
        }
        low_rows = {
            "candidate": candidate_low_full or adaptive_low,
            "backend": adaptive_low,
            "relation": adaptive_low,
            "joint": adaptive_low,
        }
        for axis in self.REQUIRED_AXES:
            comparison, result = self._finalize_axis(
                axis=axis,
                plan=normalized_plans[axis],
                low_row=low_rows[axis],
                full_row=full_rows[axis],
            )
            if comparison is not None:
                comparisons[axis] = comparison
            sentinel_results[axis] = result

        same_candidate = bool(
            full_selection is not None
            and full_selection.candidate_id == low_selection.candidate_id
        )
        naive_candidate_executions = (
            (1 if same_candidate else 2) if candidate_selected else 0
        )
        naive_audit_executions = naive_candidate_executions + sum(
            int(normalized_plans[axis].selected)
            for axis in ("backend", "relation", "joint")
        )
        actual_audit_executions = len(
            [node for node in executed_nodes if node != "adaptive_low"]
        )
        reuse_trace = {
            "schema_version": "rlcmf-coupled-factorial-reuse-trace-v1",
            "coupling_id": coupling_id,
            "design": "A-adaptive-low_B-low-candidate-full_C-full-static-v1",
            "executed_nodes": executed_nodes,
            "derived_nodes": derived_nodes,
            "node_sources": dict(sorted(node_sources.items())),
            "axis_full_row_source": {
                "candidate": (
                    "full_static_shadow" if candidate_selected else "unobserved"
                ),
                "backend": (
                    "backend_full" if normalized_plans["backend"].selected else "unobserved"
                ),
                "relation": (
                    "relation_full" if normalized_plans["relation"].selected else "unobserved"
                ),
                "joint": (
                    "full_static_shadow" if normalized_plans["joint"].selected else "unobserved"
                ),
            },
            "naive_selected_audit_execution_count": naive_audit_executions,
            "actual_selected_audit_execution_count": actual_audit_executions,
            "duplicate_audit_executions_avoided": max(
                0,
                naive_audit_executions - actual_audit_executions,
            ),
            "joint_reference_reused_from_candidate_full_shadow": bool(
                normalized_plans["joint"].selected
            ),
            "component_factorial_derivation_used": bool(derived_nodes),
        }
        return RLCMFCoupledAuditResult(
            coupling_id=coupling_id,
            low_batch=batch,
            common_batch=common_batch,
            low_selection=low_selection,
            full_selection=full_selection,
            rows=rows,
            comparisons=comparisons,
            sentinel_results=sentinel_results,
            reuse_trace=reuse_trace,
        )


@dataclass(slots=True)
class RLCMFAuditedCaseResult:
    low_row: dict[str, Any]
    audit_rows: dict[str, dict[str, Any]]
    comparisons: dict[str, CounterfactualCandidateComparison]
    sentinel_results: dict[str, SentinelAuditResult]
    audit_arm_metadata: dict[str, dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        executed_axes = [
            axis
            for axis, metadata in self.audit_arm_metadata.items()
            if metadata.get("executed") is True
        ]
        audit_wall_ms = sum(
            float(self.audit_rows[axis].get("duration_ms", 0.0) or 0.0)
            for axis in executed_axes
        )
        low_wall_ms = float(self.low_row.get("duration_ms", 0.0) or 0.0)
        audit_backend_ms = sum(
            execution_profile_backend_reported_ms(
                self.audit_rows[axis].get("execution_profile", {})
            )
            for axis in executed_axes
        )
        derivation_wall_ms = sum(
            float(metadata.get("derivation_wall_ms", 0.0) or 0.0)
            for metadata in self.audit_arm_metadata.values()
        )
        low_backend_ms = execution_profile_backend_reported_ms(
            self.low_row.get("execution_profile", {})
        )
        audit_backend_calls = sum(
            execution_profile_backend_calls(
                self.audit_rows[axis].get("execution_profile", {})
            )
            for axis in executed_axes
        )
        low_backend_calls = execution_profile_backend_calls(
            self.low_row.get("execution_profile", {})
        )
        return {
            "schema_version": "rlcmf-audited-case-summary-v1",
            "audit_execution_count": sum(
                1
                for row in self.audit_arm_metadata.values()
                if row.get("executed") is True and not row.get("reused_from_axis")
            ),
            "audit_derivation_count": sum(
                bool(row.get("derived_from_axis"))
                for row in self.audit_arm_metadata.values()
            ),
            "selected_axes": sorted(self.audit_rows),
            "low_wall_ms": low_wall_ms,
            "audit_wall_ms": audit_wall_ms + derivation_wall_ms,
            "audit_derivation_wall_ms": derivation_wall_ms,
            "combined_wall_ms": low_wall_ms + audit_wall_ms + derivation_wall_ms,
            "low_backend_reported_ms": low_backend_ms,
            "audit_backend_reported_ms": audit_backend_ms,
            "combined_backend_reported_ms": low_backend_ms + audit_backend_ms,
            "low_backend_calls": low_backend_calls,
            "audit_backend_calls": audit_backend_calls,
            "combined_backend_calls": low_backend_calls + audit_backend_calls,
            "comparisons": {
                axis: comparison.to_dict()
                for axis, comparison in sorted(self.comparisons.items())
            },
            "sentinel_results": {
                axis: result.to_dict()
                for axis, result in sorted(self.sentinel_results.items())
            },
            "audit_arms": dict(sorted(self.audit_arm_metadata.items())),
        }


class RLCMFCounterfactualExecutor:
    """Execute low fidelity first, then only preregistered selected audit arms."""

    def __init__(
        self,
        *,
        coordinator: RLCMFSentinelCoordinator,
        run_arm_fn: Callable[..., dict[str, Any]],
        event_metrics: Mapping[str, str] | None = None,
        prediction_fn: Callable[[str, Mapping[str, Any]], float] | None = None,
        factorial_deriver: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.run_arm_fn = run_arm_fn
        self.event_metrics = {
            axis: accumulator.metric
            for axis, accumulator in coordinator.event_accumulators.items()
            if axis in {"backend", "relation", "joint"}
        }
        requested_metrics = dict(event_metrics or {})
        mismatches = {
            axis: metric
            for axis, metric in requested_metrics.items()
            if self.event_metrics.get(axis) != metric
        }
        if mismatches:
            raise ValueError(
                f"counterfactual event metrics differ from frozen accumulators: {mismatches}"
            )
        if any(
            metric not in COUNTERFACTUAL_EVENT_METRICS
            for metric in self.event_metrics.values()
        ):
            raise ValueError("unknown counterfactual event metric")
        self.prediction_fn = prediction_fn or (lambda _axis, _row: 0.0)
        self.factorial_deriver = factorial_deriver

    @staticmethod
    def _config_payload(config: Any) -> Any:
        if hasattr(config, "to_dict"):
            return config.to_dict()
        if isinstance(config, Mapping):
            return dict(config)
        raise ValueError("counterfactual config must be a mapping or expose to_dict")

    def _arm_cache_key(self, backends: tuple[str, ...], config: Any) -> str:
        material = {
            "backends": list(backends),
            "config": self._config_payload(config),
        }
        return hashlib.sha256(_canonical_json_bytes(material)).hexdigest()

    def execute(
        self,
        *,
        low_backends: list[str] | tuple[str, ...],
        reference_backends: list[str] | tuple[str, ...],
        low_config: Any,
        reference_config: Any,
        plans: Mapping[str, SentinelAuditPlan],
    ) -> RLCMFAuditedCaseResult:
        normalized_plans = dict(plans)
        unsupported = sorted(set(normalized_plans) - {"backend", "relation", "joint"})
        if unsupported:
            raise ValueError(
                f"candidate/common-batch audit requires its dedicated executor: {unsupported}"
            )
        if any(axis != plan.axis for axis, plan in normalized_plans.items()):
            raise ValueError("counterfactual plan keys must match plan axes")
        low_names = tuple(dict.fromkeys(str(value) for value in low_backends))
        reference_names = tuple(
            dict.fromkeys(str(value) for value in reference_backends)
        )
        if not set(low_names).issubset(reference_names):
            raise ValueError("low-fidelity backends must be a subset of reference backends")

        low_row = self.run_arm_fn(
            arm="low",
            backends=list(low_names),
            config=low_config,
            fresh_backend_instances=False,
        )
        audit_rows: dict[str, dict[str, Any]] = {}
        comparisons: dict[str, CounterfactualCandidateComparison] = {}
        sentinel_results: dict[str, SentinelAuditResult] = {}
        audit_arm_metadata: dict[str, dict[str, Any]] = {}
        cache: dict[str, tuple[str, dict[str, Any]]] = {}
        arm_contracts = {
            "backend": (reference_names, low_config),
            "relation": (low_names, reference_config),
            "joint": (reference_names, reference_config),
        }
        execution_order = (
            ("joint", "backend", "relation")
            if self.factorial_deriver is not None
            and normalized_plans.get("joint") is not None
            and normalized_plans["joint"].selected
            else ("backend", "relation", "joint")
        )
        for axis in execution_order:
            plan = normalized_plans.get(axis)
            if plan is None:
                continue
            prediction = float(self.prediction_fn(axis, low_row))
            if not plan.selected:
                sentinel_results[axis] = self.coordinator.finalize(
                    plan,
                    outcome=None,
                    prediction=prediction,
                    full_event=None,
                    potential_misclassification=None,
                )
                audit_arm_metadata[axis] = {
                    "selected": False,
                    "executed": False,
                    "event_metric": self.event_metrics[axis],
                }
                continue
            arm_backends, arm_config = arm_contracts[axis]
            if (
                axis != "joint"
                and self.factorial_deriver is not None
                and "joint" in audit_rows
            ):
                full_row = self.factorial_deriver(
                    axis=axis,
                    joint_row=audit_rows["joint"],
                    backends=list(arm_backends),
                    config=arm_config,
                )
                comparison = CounterfactualCandidateComparison.compare(
                    low_row,
                    full_row,
                )
                outcome = comparison.event_value(self.event_metrics[axis])
                safety_kwargs: dict[str, float | None] = {}
                if axis in self.coordinator.safety_trackers:
                    safety_kwargs = {
                        "full_event": comparison.full_event_value(
                            self.event_metrics[axis]
                        ),
                        "potential_misclassification": (
                            comparison.potential_misclassification_value(
                                self.event_metrics[axis]
                            )
                        ),
                    }
                sentinel_result = self.coordinator.finalize(
                    plan,
                    outcome=outcome,
                    prediction=prediction,
                    **safety_kwargs,
                )
                audit_rows[axis] = full_row
                comparisons[axis] = comparison
                sentinel_results[axis] = sentinel_result
                audit_arm_metadata[axis] = {
                    "selected": True,
                    "executed": False,
                    "derived_from_axis": "joint",
                    "reused_from_axis": "",
                    "fresh_backend_instances_required": True,
                    "backends": list(arm_backends),
                    "config_sha256": hashlib.sha256(
                        _canonical_json_bytes(self._config_payload(arm_config))
                    ).hexdigest(),
                    "event_metric": self.event_metrics[axis],
                    "event_value": outcome,
                    "derivation_wall_ms": float(
                        full_row.get("duration_ms", 0.0) or 0.0
                    ),
                }
                continue
            cache_key = self._arm_cache_key(arm_backends, arm_config)
            reused_from_axis = ""
            if cache_key in cache:
                reused_from_axis, full_row = cache[cache_key]
            else:
                full_row = self.run_arm_fn(
                    arm=axis,
                    backends=list(arm_backends),
                    config=arm_config,
                    fresh_backend_instances=True,
                )
                cache[cache_key] = (axis, full_row)
            comparison = CounterfactualCandidateComparison.compare(low_row, full_row)
            outcome = comparison.event_value(self.event_metrics[axis])
            safety_kwargs: dict[str, float | None] = {}
            if axis in self.coordinator.safety_trackers:
                safety_kwargs = {
                    "full_event": comparison.full_event_value(
                        self.event_metrics[axis]
                    ),
                    "potential_misclassification": (
                        comparison.potential_misclassification_value(
                            self.event_metrics[axis]
                        )
                    ),
                }
            sentinel_result = self.coordinator.finalize(
                plan,
                outcome=outcome,
                prediction=prediction,
                **safety_kwargs,
            )
            audit_rows[axis] = full_row
            comparisons[axis] = comparison
            sentinel_results[axis] = sentinel_result
            audit_arm_metadata[axis] = {
                "selected": True,
                "executed": not bool(reused_from_axis),
                "reused_from_axis": reused_from_axis,
                "fresh_backend_instances_required": True,
                "backends": list(arm_backends),
                "config_sha256": hashlib.sha256(
                    _canonical_json_bytes(self._config_payload(arm_config))
                ).hexdigest(),
                "event_metric": self.event_metrics[axis],
                "event_value": outcome,
                "duration_ms": float(full_row.get("duration_ms", 0.0) or 0.0),
            }
        return RLCMFAuditedCaseResult(
            low_row=low_row,
            audit_rows=audit_rows,
            comparisons=comparisons,
            sentinel_results=sentinel_results,
            audit_arm_metadata=audit_arm_metadata,
        )
