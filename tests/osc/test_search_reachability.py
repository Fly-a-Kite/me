from __future__ import annotations

from datadiff.family_witness_registry import latest_family_witness_registrations

from datadiff_osc.generation.extraction import AtomExtractor, merge_extractions
from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.search.epochs import NoReplacementEpoch
from datadiff_osc.search.matcher import TargetMatcher
from datadiff_osc.semantic_targets.compiler import compile_target_universe
from datadiff_osc.semantic_targets.declarations import legacy_v4_target_templates
from datadiff_osc.semantic_targets.model import TargetAssignment


def test_all_232_fresh_cells_384_edges_and_502_pairs_are_deterministically_reachable():
    universe = compile_target_universe(legacy_v4_target_templates())
    registrations = {item.family_id: item for item in latest_family_witness_registrations()}
    extractor = AtomExtractor()
    matcher = TargetMatcher(universe)
    base = SeedLineage("reachability", 31000001, "lane", 0, SeedStage.TARGET)
    epoch = NoReplacementEpoch(
        (item.target_cell_id for item in universe.fresh_cells),
        base,
        namespace="fresh-cells",
    )
    cells = {item.target_cell_id: item for item in universe.fresh_cells}
    certificates = {}
    extractions = {}
    for decision_index in range(len(cells)):
        selection = epoch.select(decision_index)
        cell = cells[selection.item_id]
        case = registrations[cell.test_family_id].generate_case(cell.construction_index)
        # Preserve only physical layout metadata; assignment/family activation
        # annotations are deliberately removed before extraction.
        layouts = (case.metadata or {}).get("input_layouts")
        case.metadata = {} if layouts is None else {"input_layouts": layouts}
        assignment = TargetAssignment((cell.target_cell_id,), selection.seed_lineage)
        extraction = extractor.extract(case)
        certificate = matcher.match(assignment, extraction)
        assert certificate.valid, (cell.test_family_id, cell.coordinate_map, certificate.missing_atoms)
        certificates[cell.target_cell_id] = certificate
        extractions[cell.target_cell_id] = extraction

    assert len(certificates) == 232
    assert len(universe.fresh_edges) == 384
    edge_by_id = {item.contrast_edge_id: item for item in universe.fresh_edges}
    edge_epoch = NoReplacementEpoch(edge_by_id, base, namespace="fresh-edges")
    activated_edges = set()
    for decision_index in range(384):
        selection = edge_epoch.select(decision_index)
        edge = edge_by_id[selection.item_id]
        assignment = TargetAssignment(
            (edge.base_cell_id, edge.sibling_cell_id),
            selection.seed_lineage,
            selected_edge_ids=(edge.contrast_edge_id,),
        )
        certificate = matcher.match(
            assignment,
            merge_extractions(
                (
                    (edge.base_cell_id, extractions[edge.base_cell_id]),
                    (edge.sibling_cell_id, extractions[edge.sibling_cell_id]),
                )
            ),
        )
        assert certificate.valid
        assert set(certificate.activated_cell_ids) == {
            edge.base_cell_id,
            edge.sibling_cell_id,
        }
        activated_edges.add(edge.contrast_edge_id)
    assert len(activated_edges) == 384
    assert len(universe.fresh_backend_pair_obligations) == 502
    assert all(
        obligation.target_cell_id in certificates
        for obligation in universe.fresh_backend_pair_obligations
    )
    pair_epoch = NoReplacementEpoch(
        (item.obligation_id for item in universe.fresh_backend_pair_obligations),
        base,
        namespace="fresh-pairs",
    )
    assert len({pair_epoch.select(index).item_id for index in range(502)}) == 502
