"""Deterministic exact four-level coverage ledger."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from datadiff_osc.contract_engine.applicability import (
    ApplicabilityCertificate,
    applicability_binding_errors,
)
from datadiff_osc.contract_engine.evidence import ObservationCertificate
from datadiff_osc.contract_engine.model import Endpoint, HyperContract
from datadiff_osc.generation.extraction import ContrastExtraction, ExtractionResult
from datadiff_osc.probe_contracts import (
    ObservationPolicyError,
    materialize_coverage_hypercontract,
)
from datadiff_osc.schemas import (
    CoverageLevel,
    ExecutionStatus,
    FailureKind,
    LedgerEvent,
    StructuredExecutionOutcome,
    VerdictKind,
)
from datadiff_osc.search.matcher import ExtractionEvidence, TargetMatcher
from datadiff_osc.semantic_targets.model import (
    ActivationCertificate,
    CompiledTargetUniverse,
    TargetAssignment,
)


_LEVEL_ORDER = {
    CoverageLevel.CONSTRUCTED: 0,
    CoverageLevel.ACTIVATED: 1,
    CoverageLevel.EXECUTED: 2,
    CoverageLevel.OBSERVED: 3,
}


@dataclass(frozen=True, slots=True)
class CoverageCounts:
    level: CoverageLevel
    cells: int
    edges: int
    backend_pairs: int


class CoverageLedger:
    def __init__(self, universe: CompiledTargetUniverse) -> None:
        self.universe = universe
        self._cell_ids = {
            item.target_cell_id
            for item in (*universe.fresh_cells, *universe.regression_cells)
        }
        self._edge_ids = {item.contrast_edge_id for item in universe.fresh_edges}
        self._edge_cells = {
            item.contrast_edge_id: frozenset(
                {item.base_cell_id, item.sibling_cell_id}
            )
            for item in universe.fresh_edges
        }
        self._pair_ids = {
            item.obligation_id
            for item in (
                *universe.fresh_backend_pair_obligations,
                *universe.regression_backend_pair_obligations,
            )
        }
        self._pair_cells = {
            item.obligation_id: item.target_cell_id
            for item in (
                *universe.fresh_backend_pair_obligations,
                *universe.regression_backend_pair_obligations,
            )
        }
        self._events: dict[str, LedgerEvent] = {}
        self._matcher = TargetMatcher(universe)
        self._credits: dict[CoverageLevel, dict[str, set[str]]] = {
            level: {"cells": set(), "edges": set(), "pairs": set()}
            for level in CoverageLevel
        }

    @property
    def events(self) -> tuple[LedgerEvent, ...]:
        return tuple(
            sorted(
                self._events.values(),
                key=lambda item: (item.event_order_key, item.event_id),
            )
        )

    def _validate_targets(self, event: LedgerEvent) -> None:
        if not (event.cell_ids or event.edge_ids or event.backend_pair_obligation_ids):
            raise ValueError("coverage event must name a cell, edge or backend pair")
        unknown_cells = set(event.cell_ids) - self._cell_ids
        unknown_edges = set(event.edge_ids) - self._edge_ids
        unknown_pairs = set(event.backend_pair_obligation_ids) - self._pair_ids
        if unknown_cells or unknown_edges or unknown_pairs:
            raise ValueError(
                "coverage event contains unknown targets: "
                f"cells={sorted(unknown_cells)} edges={sorted(unknown_edges)} "
                f"pairs={sorted(unknown_pairs)}"
            )

    def _validate_monotonic(self, event: LedgerEvent) -> None:
        index = _LEVEL_ORDER[event.level]
        if index == 0:
            return
        previous = tuple(CoverageLevel)[index - 1]
        checks = (
            ("cells", event.cell_ids),
            ("edges", event.edge_ids),
            ("pairs", event.backend_pair_obligation_ids),
        )
        for kind, identities in checks:
            missing = set(identities) - self._credits[previous][kind]
            if missing:
                raise ValueError(
                    f"coverage level {event.level.value} lacks prior "
                    f"{previous.value} credit: {sorted(missing)}"
                )

    def _validate_activation_evidence(
        self,
        event: LedgerEvent,
        *,
        target_assignment: TargetAssignment | None,
        extraction: ExtractionEvidence | None,
        activation_certificate: ActivationCertificate | None,
    ) -> ActivationCertificate:
        if not isinstance(target_assignment, TargetAssignment):
            raise ValueError("coverage activation requires a real TargetAssignment")
        if not isinstance(extraction, (ExtractionResult, ContrastExtraction)):
            raise ValueError("coverage activation requires real extraction evidence")
        if not isinstance(activation_certificate, ActivationCertificate):
            raise ValueError("coverage activation requires a real certificate")
        if event.activation_certificate_digest != activation_certificate.digest:
            raise ValueError("activation certificate digest mismatch")
        recomputed = self._matcher.match(
            target_assignment,
            extraction,
            preflight_valid=activation_certificate.preflight_valid,
            mutation_preserved=activation_certificate.mutation_preserved,
            degraded_reasons=activation_certificate.degraded_reasons,
        )
        if recomputed != activation_certificate:
            raise ValueError(
                "activation certificate does not match universe, assignment, or atoms"
            )
        if not activation_certificate.valid:
            raise ValueError("invalid activation certificate cannot earn coverage")

        if not set(event.edge_ids) <= set(target_assignment.selected_edge_ids):
            raise ValueError("credited edge is not bound to the target assignment")
        required_cells = set(event.cell_ids)
        required_cells.update(
            cell_id
            for edge_id in event.edge_ids
            for cell_id in self._edge_cells[edge_id]
        )
        required_cells.update(
            self._pair_cells[pair_id]
            for pair_id in event.backend_pair_obligation_ids
        )
        if not required_cells <= set(target_assignment.selected_cell_ids):
            raise ValueError("coverage targets are not selected by the assignment")
        if not required_cells <= set(activation_certificate.activated_cell_ids):
            raise ValueError("activation certificate does not bind credited targets")
        return activation_certificate

    def _validate_execution_evidence(
        self,
        event: LedgerEvent,
        *,
        activation_certificate: ActivationCertificate,
        extraction: ExtractionEvidence,
        contract: HyperContract | None,
        endpoints: tuple[Endpoint, ...] | None,
        applicability_certificate: ApplicabilityCertificate | None,
        execution_outcomes: tuple[StructuredExecutionOutcome, ...] | None,
    ) -> tuple[
        HyperContract,
        tuple[Endpoint, ...],
        ApplicabilityCertificate,
        tuple[StructuredExecutionOutcome, ...],
    ]:
        if not isinstance(contract, HyperContract):
            raise ValueError("executed coverage requires a real HyperContract")
        if (
            not isinstance(endpoints, tuple)
            or not endpoints
            or any(not isinstance(item, Endpoint) for item in endpoints)
        ):
            raise ValueError("executed coverage requires the original Endpoint tuple")
        try:
            expected_contract = materialize_coverage_hypercontract(
                self.universe,
                cell_ids=event.cell_ids,
                edge_ids=event.edge_ids,
                backend_pair_obligation_ids=event.backend_pair_obligation_ids,
                extraction=extraction,
                endpoints=endpoints,
            )
        except ObservationPolicyError as exc:
            raise ValueError(
                f"coverage observation policy rejected authority: {exc}"
            ) from exc
        if contract != expected_contract:
            raise ValueError(
                "coverage contract does not equal the unique materialized "
                "HyperContract"
            )
        if not isinstance(applicability_certificate, ApplicabilityCertificate):
            raise ValueError(
                "executed coverage requires a real ApplicabilityCertificate"
            )
        if event.applicability_certificate_digest != applicability_certificate.digest:
            raise ValueError("applicability certificate digest mismatch")
        binding_errors = applicability_binding_errors(
            contract, endpoints, applicability_certificate
        )
        if binding_errors:
            raise ValueError(
                "applicability certificate is not bound to original endpoints: "
                + ";".join(binding_errors)
            )
        if not applicability_certificate.valid:
            raise ValueError("non-applicable execution cannot earn coverage credit")
        observed_atoms = set(activation_certificate.observed_atoms)
        if not set(applicability_certificate.selected_atoms) <= observed_atoms:
            raise ValueError("applicability selected atoms are not activation-bound")
        if not set(applicability_certificate.activated_atoms) <= observed_atoms:
            raise ValueError("applicability activated atoms are not activation-bound")

        if (
            not isinstance(execution_outcomes, tuple)
            or not execution_outcomes
            or any(
                not isinstance(item, StructuredExecutionOutcome)
                for item in execution_outcomes
            )
        ):
            raise ValueError(
                "executed coverage requires real StructuredExecutionOutcome objects"
            )
        endpoint_ids = tuple(item.endpoint_id for item in endpoints)
        outcome_ids = tuple(item.endpoint_id for item in execution_outcomes)
        if outcome_ids != endpoint_ids:
            raise ValueError("execution outcomes do not match original endpoints")
        if event.execution_outcome_digests != tuple(
            item.digest for item in execution_outcomes
        ):
            raise ValueError("execution outcome digest mismatch")
        if any(
            item.status is not ExecutionStatus.OK
            or item.failure_kind is not FailureKind.NONE
            for item in execution_outcomes
        ):
            raise ValueError("non-OK execution cannot earn executed coverage credit")
        return contract, endpoints, applicability_certificate, execution_outcomes

    @staticmethod
    def _validate_observation_evidence(
        event: LedgerEvent,
        *,
        contract: HyperContract,
        endpoints: tuple[Endpoint, ...],
        applicability_certificate: ApplicabilityCertificate,
        observation_certificate: ObservationCertificate | None,
    ) -> None:
        if not isinstance(observation_certificate, ObservationCertificate):
            raise ValueError(
                "observed coverage requires a real ObservationCertificate"
            )
        if event.observation_certificate_digest != observation_certificate.digest:
            raise ValueError("observation certificate digest mismatch")
        endpoint_ids = tuple(item.endpoint_id for item in endpoints)
        binding_errors = observation_certificate.binding_errors(
            contract_digest=contract.digest,
            applicability_digest=applicability_certificate.digest,
            endpoint_ids=endpoint_ids,
        )
        if binding_errors:
            raise ValueError(
                "observation certificate is not authority-bound: "
                + ";".join(binding_errors)
            )
        if not observation_certificate.observer_ids:
            raise ValueError("observation authority did not evaluate an observer")
        if observation_certificate.comparison_stage == "S0_STATIC":
            raise ValueError("S0 applicability evidence is not observed coverage")
        if observation_certificate.verdict.kind not in {
            VerdictKind.SATISFIED,
            VerdictKind.VIOLATED,
        }:
            raise ValueError(
                "inapplicable or inconclusive observation cannot earn coverage"
            )
        if (
            observation_certificate.verdict.kind is VerdictKind.VIOLATED
            and not observation_certificate.exact_escalated
        ):
            raise ValueError("violated observation requires exact authority")

    def admit(
        self,
        event: LedgerEvent,
        *,
        target_assignment: TargetAssignment | None = None,
        extraction: ExtractionEvidence | None = None,
        activation_certificate: ActivationCertificate | None = None,
        contract: HyperContract | None = None,
        endpoints: tuple[Endpoint, ...] | None = None,
        applicability_certificate: ApplicabilityCertificate | None = None,
        observation_certificate: ObservationCertificate | None = None,
        execution_outcomes: tuple[StructuredExecutionOutcome, ...] | None = None,
    ) -> bool:
        self._validate_targets(event)
        if event.event_id in self._events:
            return False
        validated_activation: ActivationCertificate | None = None
        if event.level != CoverageLevel.CONSTRUCTED:
            validated_activation = self._validate_activation_evidence(
                event,
                target_assignment=target_assignment,
                extraction=extraction,
                activation_certificate=activation_certificate,
            )
        validated_execution = None
        if event.level in {CoverageLevel.EXECUTED, CoverageLevel.OBSERVED}:
            assert validated_activation is not None
            validated_execution = self._validate_execution_evidence(
                event,
                activation_certificate=validated_activation,
                extraction=extraction,
                contract=contract,
                endpoints=endpoints,
                applicability_certificate=applicability_certificate,
                execution_outcomes=execution_outcomes,
            )
        if event.level == CoverageLevel.OBSERVED:
            assert validated_execution is not None
            validated_contract, validated_endpoints, validated_applicability, _ = (
                validated_execution
            )
            self._validate_observation_evidence(
                event,
                contract=validated_contract,
                endpoints=validated_endpoints,
                applicability_certificate=validated_applicability,
                observation_certificate=observation_certificate,
            )
        self._admit_frozen(event)
        return True

    def _admit_frozen(self, event: LedgerEvent) -> None:
        self._validate_targets(event)
        self._validate_monotonic(event)
        self._events[event.event_id] = event
        self._credits[event.level]["cells"].update(event.cell_ids)
        self._credits[event.level]["edges"].update(event.edge_ids)
        self._credits[event.level]["pairs"].update(event.backend_pair_obligation_ids)

    def counts(self, level: CoverageLevel) -> CoverageCounts:
        values = self._credits[level]
        return CoverageCounts(
            level=level,
            cells=len(values["cells"]),
            edges=len(values["edges"]),
            backend_pairs=len(values["pairs"]),
        )

    def credited_ids(self, level: CoverageLevel, kind: str) -> frozenset[str]:
        if kind not in {"cells", "edges", "pairs"}:
            raise ValueError(f"unknown coverage kind: {kind}")
        return frozenset(self._credits[level][kind])

    def family_projection(self, level: CoverageLevel) -> dict[str, tuple[int, int]]:
        credited = self._credits[level]["cells"]
        totals: dict[str, int] = defaultdict(int)
        hits: dict[str, int] = defaultdict(int)
        for cell in (*self.universe.fresh_cells, *self.universe.regression_cells):
            totals[cell.test_family_id] += 1
            hits[cell.test_family_id] += int(cell.target_cell_id in credited)
        return {family: (hits[family], total) for family, total in sorted(totals.items())}

    @classmethod
    def merge(
        cls,
        universe: CompiledTargetUniverse,
        ledgers: Iterable["CoverageLedger"],
    ) -> "CoverageLedger":
        materialized = tuple(ledgers)
        mismatched = [
            ledger.universe.digest
            for ledger in materialized
            if ledger.universe.digest != universe.digest
        ]
        if mismatched:
            raise ValueError("cannot merge coverage ledgers from another universe")
        events = {
            event.event_id: event
            for ledger in materialized
            for event in ledger.events
        }
        merged = cls(universe)
        for event in sorted(events.values(), key=lambda item: (item.event_order_key, item.event_id)):
            merged._admit_frozen(event)
        return merged


__all__ = ["CoverageCounts", "CoverageLedger"]
