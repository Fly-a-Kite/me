from __future__ import annotations

from datadiff.dsl import Case, ColumnSpec, Program, TableData

from datadiff_osc.generation.extraction import AtomExtractor, merge_extractions


def _case(metadata=None):
    return Case(
        case_id="extract",
        seed=1,
        tables=[
            TableData(
                name="t0",
                columns=[
                    ColumnSpec("id", "int", False),
                    ColumnSpec("g", "str", False),
                    ColumnSpec("x", "float", True),
                ],
                rows=[
                    {"id": 1, "g": "a", "x": -1.0},
                    {"id": 2, "g": "a", "x": 0.0},
                    {"id": 3, "g": "b", "x": 2.0},
                    {"id": 4, "g": "b", "x": None},
                ],
            )
        ],
        program=Program(
            program_id="p",
            seed=1,
            operations=[
                {"op": "filter", "column": "x", "cmp": ">=", "value": 0.0},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                },
            ],
        ),
        metadata=dict(metadata or {}),
    )


def test_extraction_has_provenance_and_observable_program_data_atoms():
    result = AtomExtractor().extract(_case())
    assert {"op:filter", "op:groupby", "agg:sum", "data:null_values"} <= result.atom_ids
    assert all(atom.provenance for atom in result.atoms)
    assert {item.source_kind for atom in result.atoms for item in atom.provenance} <= {
        "ccs_ir",
        "input_layout",
        "table_data",
    }


def test_assignment_or_family_metadata_cannot_change_atoms_and_cache_is_single_pass():
    extractor = AtomExtractor()
    first = extractor.extract(_case({"target_assignment": {"cell": "forged"}}))
    second = extractor.extract(_case({"family_witness": {"semantic_activation": True}}))
    assert first is second
    assert extractor.extraction_count == 1
    assert first.atom_ids == second.atom_ids


def test_data_change_invalidates_cache_and_changes_data_digest():
    extractor = AtomExtractor()
    first = extractor.extract(_case())
    changed = _case()
    changed.tables[0].rows[-1]["x"] = 9.0
    second = extractor.extract(changed)
    assert extractor.extraction_count == 2
    assert first.data_digest != second.data_digest


def test_contrast_merge_binds_endpoint_direction_instead_of_sorting_it_away():
    extractor = AtomExtractor()
    base = extractor.extract(_case())
    sibling_case = _case()
    sibling_case.tables[0].rows[-1]["x"] = 9.0
    sibling = extractor.extract(sibling_case)

    forward = merge_extractions((("base-cell", base), ("sibling-cell", sibling)))
    swapped = merge_extractions((("sibling-cell", sibling), ("base-cell", base)))

    assert forward.digest != swapped.digest
    assert forward.endpoint_ids == ("base-cell", "sibling-cell")
    assert forward.extraction_for("base-cell") == base


def test_contrast_merge_rejects_anonymous_duplicate_or_one_sided_evidence():
    extraction = AtomExtractor().extract(_case())
    import pytest

    with pytest.raises(TypeError, match="target_cell_id"):
        merge_extractions((extraction, extraction))
    with pytest.raises(ValueError, match="at least two"):
        merge_extractions((("only", extraction),))
    with pytest.raises(ValueError, match="unique"):
        merge_extractions((("same", extraction), ("same", extraction)))
