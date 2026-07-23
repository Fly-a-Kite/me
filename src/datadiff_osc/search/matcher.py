"""Exact all/any/none target matching over provenance-bearing atoms."""

from __future__ import annotations

from dataclasses import dataclass

from datadiff_osc._canonical import stable_digest
from datadiff_osc.schemas import TargetFingerprint
from datadiff_osc.semantic_targets.model import (
    ActivationCertificate,
    CompiledTargetUniverse,
    ContrastEdge,
    TargetAssignment,
    TargetCell,
)
from datadiff_osc.generation.extraction import ContrastExtraction, ExtractionResult


ExtractionEvidence = ExtractionResult | ContrastExtraction


_FORBIDDEN_EVIDENCE_SOURCES = frozenset(
    {
        "target_assignment",
        "case_metadata",
        "family_metadata",
        "activation_metadata",
    }
)


@dataclass(frozen=True, slots=True)
class CellMatch:
    target_cell_id: str
    matched: bool
    missing_all: tuple[str, ...]
    missing_any_groups: tuple[tuple[str, ...], ...]
    observed_forbidden: tuple[str, ...]


class TargetMatcher:
    def __init__(self, universe: CompiledTargetUniverse) -> None:
        self.universe = universe
        self.target_fingerprint = TargetFingerprint(
            universe_digest=universe.digest,
            taxonomy_digest=universe.taxonomy_digest,
            template_digest=universe.template_digest,
        )
        self._cells = {
            item.target_cell_id: item
            for item in (*universe.fresh_cells, *universe.regression_cells)
        }
        self._edges = {
            item.contrast_edge_id: item for item in universe.fresh_edges
        }

    @staticmethod
    def _reject_circular_evidence(extraction: ExtractionEvidence) -> None:
        for atom in extraction.atoms:
            if not atom.provenance:
                raise ValueError("matcher requires provenance for every observed atom")
            for provenance in atom.provenance:
                if provenance.source_kind in _FORBIDDEN_EVIDENCE_SOURCES:
                    raise ValueError(
                        "target assignment or metadata cannot provide activation evidence"
                    )

    @staticmethod
    def match_cell(cell: TargetCell, observed: frozenset[str]) -> CellMatch:
        missing_all = tuple(sorted(cell.required_all_atoms - observed))
        missing_groups = tuple(
            tuple(sorted(group))
            for group in cell.required_any_atom_groups
            if not group & observed
        )
        forbidden = tuple(sorted(cell.forbidden_atoms & observed))
        return CellMatch(
            target_cell_id=cell.target_cell_id,
            matched=not missing_all and not missing_groups and not forbidden,
            missing_all=missing_all,
            missing_any_groups=missing_groups,
            observed_forbidden=forbidden,
        )

    def _edge_evidence(
        self,
        assignment: TargetAssignment,
        extraction: ExtractionEvidence,
    ) -> tuple[dict[str, frozenset[str]], tuple[str, ...]]:
        edge_ids = tuple(assignment.selected_edge_ids)
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("target assignment must select unique contrast edges")
        unknown_edges = set(edge_ids) - set(self._edges)
        if unknown_edges:
            raise ValueError(
                f"target assignment contains unknown edges: {sorted(unknown_edges)}"
            )
        if not edge_ids:
            if isinstance(extraction, ContrastExtraction):
                raise ValueError(
                    "endpoint-bound contrast evidence requires selected edge IDs"
                )
            return (
                {
                    cell_id: extraction.atom_ids
                    for cell_id in assignment.selected_cell_ids
                },
                (),
            )
        if not isinstance(extraction, ContrastExtraction):
            raise ValueError(
                "contrast edge activation requires endpoint-bound extraction evidence"
            )

        expected_endpoint_order: list[str] = []
        selected_edges: list[ContrastEdge] = []
        for edge_id in edge_ids:
            edge = self._edges[edge_id]
            selected_edges.append(edge)
            for cell_id in (edge.base_cell_id, edge.sibling_cell_id):
                if cell_id not in expected_endpoint_order:
                    expected_endpoint_order.append(cell_id)
        expected = tuple(expected_endpoint_order)
        if tuple(assignment.selected_cell_ids) != expected:
            raise ValueError(
                "contrast assignment cell order must be base then sibling for each edge"
            )
        if extraction.endpoint_ids != expected:
            raise ValueError(
                "contrast extraction endpoint order does not match edge direction"
            )

        observed_by_cell = {
            cell_id: extraction.extraction_for(cell_id).atom_ids
            for cell_id in expected
        }
        binding_digests = tuple(
            stable_digest(
                "osc-contrast-edge-activation-binding",
                {
                    "target_fingerprint": self.target_fingerprint,
                    "assignment_digest": assignment.digest,
                    "seed_lineage_digest": assignment.seed_lineage.digest,
                    "edge": edge,
                    "base_extraction_digest": extraction.extraction_for(
                        edge.base_cell_id
                    ).digest,
                    "sibling_extraction_digest": extraction.extraction_for(
                        edge.sibling_cell_id
                    ).digest,
                },
            )
            for edge in selected_edges
        )
        return observed_by_cell, binding_digests

    def match(
        self,
        assignment: TargetAssignment,
        extraction: ExtractionEvidence,
        *,
        preflight_valid: bool = True,
        mutation_preserved: bool = True,
        degraded_reasons: tuple[str, ...] = (),
    ) -> ActivationCertificate:
        self._reject_circular_evidence(extraction)
        selected = tuple(assignment.selected_cell_ids)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("target assignment must select unique cells")
        unknown = set(selected) - set(self._cells)
        if unknown:
            raise ValueError(f"target assignment contains unknown cells: {sorted(unknown)}")

        observed_by_cell, edge_binding_digests = self._edge_evidence(
            assignment, extraction
        )
        observed = frozenset(
            atom_id
            for cell_atoms in observed_by_cell.values()
            for atom_id in cell_atoms
        )
        matches = tuple(
            self.match_cell(self._cells[cell_id], observed_by_cell[cell_id])
            for cell_id in selected
        )
        activated = tuple(sorted(item.target_cell_id for item in matches if item.matched))
        required = tuple(
            sorted(
                {
                    atom
                    for cell_id in selected
                    for atom in (
                        *self._cells[cell_id].required_all_atoms,
                        *(
                            candidate
                            for group in self._cells[cell_id].required_any_atom_groups
                            for candidate in group
                        ),
                    )
                }
            )
        )
        missing = tuple(
            sorted(
                {
                    *(
                        atom
                        for item in matches
                        for atom in item.missing_all
                    ),
                    *(
                        "any:" + "|".join(group)
                        for item in matches
                        for group in item.missing_any_groups
                    ),
                }
            )
        )
        forbidden = tuple(
            sorted(
                {
                    atom
                    for item in matches
                    for atom in item.observed_forbidden
                }
            )
        )
        return ActivationCertificate(
            target_fingerprint=self.target_fingerprint,
            assignment_digest=assignment.digest,
            selected_cell_ids=tuple(sorted(selected)),
            activated_cell_ids=activated,
            required_atoms=required,
            observed_atoms=tuple(sorted(observed)),
            missing_atoms=missing,
            forbidden_atoms=forbidden,
            static_facts=tuple(
                sorted(
                    {
                        *extraction.static_facts,
                        *(
                            f"contrast_edge_binding:{digest}"
                            for digest in edge_binding_digests
                        ),
                    }
                )
            ),
            semantic_atom_digests=tuple(
                sorted(
                    {
                        *(item.digest for item in extraction.atoms),
                        *edge_binding_digests,
                    }
                )
            ),
            preflight_valid=bool(preflight_valid),
            mutation_preserved=bool(mutation_preserved),
            degraded_reasons=tuple(degraded_reasons),
        )


__all__ = ["CellMatch", "ExtractionEvidence", "TargetMatcher"]
