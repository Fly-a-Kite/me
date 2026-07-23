"""Target-preserving mutation with mandatory re-extraction and exact matching."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

from datadiff.dsl import Case, Program, TableData

from datadiff_osc.generation.extraction import AtomExtractor
from datadiff_osc.schemas import SeedStage
from datadiff_osc.search.epochs import derive_stage_lineage
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.model import ActivationCertificate, TargetAssignment


MutationOperator = Callable[[Case, int], Case]


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    accepted_case: Case | None
    certificate: ActivationCertificate
    status: str
    reason: str
    attempts: int
    mutation_seed: int

    @property
    def accepted(self) -> bool:
        return self.accepted_case is not None and self.certificate.valid


def _clone_case(case: Case) -> Case:
    return Case.from_dict(copy.deepcopy(case.to_dict()))


def _repair_atom_classes(
    original: Case,
    candidate: Case,
    missing_atoms: tuple[str, ...],
) -> Case:
    repaired = _clone_case(candidate)
    namespaces = {item.partition(":")[0] for item in missing_atoms}
    if namespaces & {
        "op",
        "expr",
        "agg",
        "chain",
        "pipeline",
        "axis",
        "order",
        "cast",
        "boundary",
    }:
        repaired.program = Program.from_dict(copy.deepcopy(original.program.to_dict()))
    if namespaces & {"data", "rows", "type", "partition"}:
        repaired.tables = [TableData.from_dict(copy.deepcopy(item.to_dict())) for item in original.tables]
    if namespaces & {"layout", "mode"}:
        repaired.metadata = dict(repaired.metadata or {})
        original_layouts = (original.metadata or {}).get("input_layouts")
        if original_layouts is None:
            repaired.metadata.pop("input_layouts", None)
        else:
            repaired.metadata["input_layouts"] = copy.deepcopy(original_layouts)
    return repaired


class TargetPreservingMutator:
    def __init__(
        self,
        matcher: TargetMatcher,
        *,
        extractor: AtomExtractor | None = None,
    ) -> None:
        self.matcher = matcher
        self.extractor = extractor or AtomExtractor()

    def mutate(
        self,
        case: Case,
        assignment: TargetAssignment,
        operator: MutationOperator,
        *,
        allow_repair: bool = True,
    ) -> MutationOutcome:
        lineage = derive_stage_lineage(assignment.seed_lineage, SeedStage.MUTATION)
        original = _clone_case(case)
        original_extraction = self.extractor.extract(original)
        candidate = operator(_clone_case(case), lineage.subseed)
        if not isinstance(candidate, Case):
            raise TypeError("mutation operator must return a Case")
        extraction = self.extractor.extract(candidate)
        certificate = self.matcher.match(assignment, extraction, mutation_preserved=True)
        if certificate.valid and extraction.source_digest != original_extraction.source_digest:
            return MutationOutcome(candidate, certificate, "accepted", "target_preserved", 1, lineage.subseed)
        repair_atoms = (*certificate.missing_atoms, *certificate.forbidden_atoms)
        if allow_repair and repair_atoms:
            repaired = _repair_atom_classes(original, candidate, repair_atoms)
            repaired_extraction = self.extractor.extract(repaired)
            repaired_certificate = self.matcher.match(
                assignment, repaired_extraction, mutation_preserved=True
            )
            if (
                repaired_certificate.valid
                and repaired_extraction.source_digest != original_extraction.source_digest
            ):
                return MutationOutcome(
                    repaired,
                    repaired_certificate,
                    "repaired",
                    "generic_atom_class_repair",
                    2,
                    lineage.subseed,
                )
            certificate = repaired_certificate
        rejected = self.matcher.match(
            assignment,
            self.extractor.extract(candidate),
            mutation_preserved=False,
            degraded_reasons=("mutation_target_not_preserved",),
        )
        return MutationOutcome(
            None,
            rejected,
            "rejected",
            "target_not_preserved_or_semantic_noop",
            2 if allow_repair else 1,
            lineage.subseed,
        )


__all__ = ["MutationOperator", "MutationOutcome", "TargetPreservingMutator"]
