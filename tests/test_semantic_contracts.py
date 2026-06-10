from datadiff.classification_oracle import classify_finding
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.semantic_contracts import (
    finding_contract_axes,
    finding_matches_contract_boundary,
    semantic_contract_lattice_payload,
)


def test_semantic_contract_lattice_declares_boundary_axes_for_operations():
    case = Case(
        "case-contract",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("s", "str"), ColumnSpec("x", "float")],
                [{"s": "1", "x": float("nan")}, {"s": None, "x": 2.0}],
            )
        ],
        Program(
            "prog-contract",
            1,
            [
                {"op": "mutate", "column": "sx", "expr": {"kind": "cast", "source": "s", "to": "int"}},
                {"op": "filter", "column": "sx", "cmp": ">=", "value": None},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    payload = semantic_contract_lattice_payload(case)

    assert payload["schema_version"] == "semantic-contract-lattice-v1"
    assert payload["axes"]["dtype_coercion"]["policy"] == "boundary"
    assert payload["axes"]["error_equivalence"]["policy"] == "boundary"
    assert payload["axes"]["null"]["policy"] == "boundary"
    assert payload["axes"]["nan"]["policy"] == "boundary"
    assert payload["axes"]["ordering"]["policy"] == "boundary"
    assert "dtype_coercion" in payload["boundary_axes"]


def test_finding_contract_boundary_only_matches_declared_boundary_axes():
    boundary_case = Case(
        "case-cast-boundary",
        2,
        [TableData("t0", [ColumnSpec("s", "str")], [{"s": "1"}, {"s": "bad"}])],
        Program("prog-cast-boundary", 2, [{"op": "mutate", "column": "i", "expr": {"kind": "cast", "source": "s", "to": "int"}}]),
    )
    strict_case = Case(
        "case-strict",
        3,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-strict", 3, [{"op": "select", "columns": ["x"]}]),
    )
    finding = {"root_cause": "type_cast", "mismatch_class": "status"}

    assert finding_contract_axes(finding) == ("dtype_coercion", "error_equivalence")
    assert finding_matches_contract_boundary(boundary_case, finding)
    assert not finding_matches_contract_boundary(strict_case, finding)


def test_classification_uses_semantic_contract_lattice_boundary():
    case = Case(
        "case-contract-classification",
        4,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])],
        Program("prog-contract-classification", 4, [{"op": "mutate", "column": "xf", "expr": {"kind": "cast", "source": "x", "to": "float"}}]),
    )
    finding = {
        "kind": "semantic_output_mismatch",
        "root_cause": "type_cast",
        "mismatch_class": "status",
        "confidence": "high",
        "suspicious_backends": ["duckdb"],
    }

    classification = classify_finding(case, finding, {}, {}, {"generator_profile": "common"}, ["pandas", "duckdb"])

    assert classification.verdict == "expected_semantic_divergence"
    assert "semantic-contract axis" in classification.evidence
