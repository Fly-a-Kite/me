"""Boundary rule for filters that compare against NaN/Infinity.

PyArrow follows IEEE ordering (0 <= NaN is False) while the reference and the
other backends treat NaN as nullish (unknown), so such cases must be classified
as expected semantic divergence rather than an implementation bug.
"""

from __future__ import annotations

from datadiff.case_features import case_has_special_float_filter_literal
from datadiff.classification_oracle import SEMANTIC_BOUNDARY_RULES
from datadiff.dsl import Case


def _case_with_filter_value(value: object) -> Case:
    return Case.from_dict(
        {
            "case_id": "special-float-filter",
            "seed": 1,
            "tables": [
                {
                    "name": "t0",
                    "columns": [{"name": "x", "type": "int", "nullable": True}],
                    "rows": [{"x": 0}],
                }
            ],
            "program": {
                "program_id": "p",
                "seed": 1,
                "operations": [
                    {"op": "filter", "column": "x", "cmp": "le_is_not_false", "value": value}
                ],
            },
        }
    )


def test_nan_filter_literal_is_detected() -> None:
    assert case_has_special_float_filter_literal(_case_with_filter_value(float("nan")))


def test_infinity_filter_literal_is_detected() -> None:
    assert case_has_special_float_filter_literal(_case_with_filter_value(float("inf")))


def test_ordinary_filter_literal_is_not_special() -> None:
    assert not case_has_special_float_filter_literal(_case_with_filter_value(5))


def test_boundary_rule_matches_special_float_filter_literal() -> None:
    case = _case_with_filter_value(float("nan"))
    finding = {"kind": "semantic_output_mismatch", "root_cause": "conditional_expression"}
    matched = [rule.rule_id for rule in SEMANTIC_BOUNDARY_RULES if rule.predicate(case, finding, {})]
    assert "boundary:special_float_filter_literal" in matched
