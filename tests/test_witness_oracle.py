from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.normalizer import NormalizedResult
from datadiff.witness_oracle import (
    build_reference_witness_plan,
    evaluate_witness_contract,
    finding_from_witness_result,
    infer_reference_witness_contract,
    infer_witness_contract,
    rectify_predicate_for_row,
    row_satisfies_predicate,
    witness_contract_from_case,
)


def _case_with_contract(contract: dict) -> Case:
    return Case(
        "case-witness",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"id": 1, "g": "a", "x": 3}],
            )
        ],
        Program("prog-witness", 1, [{"op": "select", "columns": ["id", "g", "x"]}]),
        metadata={"witness_contract": contract},
    )


def test_witness_contract_from_case_is_optional_and_backward_compatible():
    case = Case(
        "case-no-witness",
        1,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog", 1, []),
    )

    assert witness_contract_from_case(case) is None
    result = evaluate_witness_contract(case, {}, enabled=False)
    assert result.to_dict()["contract_present"] is False
    assert result.to_dict()["enabled"] is False


def test_row_containment_witness_identifies_failing_backend():
    case = _case_with_contract(
        {
            "kind": "row_containment",
            "row": {"id": 1, "g": "a"},
            "reason": "pivot row must survive the selected workflow",
        }
    )
    normalized = {
        "left": NormalizedResult("left", "ok", ["id", "g", "x"], [[1, "a", 3]]),
        "right": NormalizedResult("right", "ok", ["id", "g", "x"], [[2, "b", 4]]),
    }

    result = evaluate_witness_contract(case, normalized, enabled=True)
    payload = result.to_dict()

    assert payload["passed"] is False
    assert payload["satisfied_backends"] == ["left"]
    assert payload["failing_backends"] == ["right"]
    finding = finding_from_witness_result(case, result)
    assert finding is not None
    assert finding.oracle == "witness"
    assert finding.kind == "witness_row_containment_violation"
    assert finding.suspicious_backends == ["right"]


def test_group_containment_and_aggregate_value_witnesses():
    normalized = {
        "duckdb": NormalizedResult(
            "duckdb",
            "ok",
            ["g", "count_x"],
            [["a", 2], ["b", 1]],
        ),
        "other": NormalizedResult(
            "other",
            "ok",
            ["g", "count_x"],
            [["b", 1]],
        ),
    }
    group_case = _case_with_contract({"kind": "group_containment", "group_key": {"g": "a"}})
    agg_case = _case_with_contract(
        {
            "kind": "aggregate_value",
            "group_key": {"g": "a"},
            "aggregate": "count_x",
            "expected": 2,
        }
    )

    group_result = evaluate_witness_contract(group_case, normalized, enabled=True).to_dict()
    agg_result = evaluate_witness_contract(agg_case, normalized, enabled=True).to_dict()

    assert group_result["satisfied_backends"] == ["duckdb"]
    assert group_result["failing_backends"] == ["other"]
    assert agg_result["satisfied_backends"] == ["duckdb"]
    assert agg_result["failing_backends"] == ["other"]


def test_reference_consensus_prefers_aggregate_witness_for_grouped_outputs():
    normalized = {
        "duckdb": NormalizedResult("duckdb", "ok", ["count_x", "g", "sum_x"], [[1, "a", 10]]),
        "pandas": NormalizedResult("pandas", "ok", ["count_x", "g", "sum_x"], [[1, "a", 11]]),
        "sqlite": NormalizedResult("sqlite", "ok", ["count_x", "g", "sum_x"], [[1, "a", 11]]),
    }

    contract = infer_reference_witness_contract(
        normalized,
        suspicious_backends=["duckdb"],
        reference_backends=["pandas", "sqlite"],
        operations=[
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"func": "count", "column": "x", "as": "count_x"},
                    {"func": "sum", "column": "x", "as": "sum_x"},
                ],
            }
        ],
        root_cause="groupby_aggregation",
        family="groupby_aggregation@duckdb",
    )

    assert contract is not None
    assert contract.kind == "aggregate_value"
    assert contract.group_key == {"g": "a"}
    assert contract.aggregate == "sum_x"
    assert contract.expected == 11
    assert contract.backends == ["duckdb", "pandas", "sqlite"]


def test_reference_witness_plan_records_reference_and_suspicious_outcomes():
    normalized = {
        "duckdb": {
            "status": "ok",
            "columns": ["id", "value"],
            "rows": [[1, "wrong"]],
        },
        "pandas": {
            "status": "ok",
            "columns": ["id", "value"],
            "rows": [[1, "expected"]],
        },
        "sqlite": {
            "status": "ok",
            "columns": ["id", "value"],
            "rows": [[1, "expected"]],
        },
    }

    plan = build_reference_witness_plan(
        normalized,
        suspicious_backends=["duckdb"],
        reference_backends=["pandas", "sqlite"],
        root_cause="ordering_or_limit",
        family="ordering_or_limit@duckdb",
    )

    assert plan["status"] == "available"
    assert plan["contract"]["kind"] == "row_containment"
    assert plan["contract"]["row"] == {"id": 1, "value": "expected"}
    assert plan["satisfied_reference_backends"] == ["pandas", "sqlite"]
    assert plan["failing_suspicious_backends"] == ["duckdb"]
    assert plan["policy"] == "triage_and_reproducer_only_not_authority_count"


def test_missing_column_does_not_satisfy_null_witness():
    case = _case_with_contract({"kind": "row_containment", "row": {"missing": None}})
    normalized = {
        "backend": NormalizedResult("backend", "ok", ["id"], [[1]]),
    }

    result = evaluate_witness_contract(case, normalized, enabled=True).to_dict()

    assert result["satisfied_backends"] == []
    assert result["failing_backends"] == ["backend"]


def test_infers_row_containment_for_row_preserving_pipeline_only():
    case = Case(
        "case-infer",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
                [{"id": 1, "x": 1}, {"id": 2, "x": 3}],
            )
        ],
        Program(
            "prog-infer",
            1,
            [
                {"op": "filter", "column": "x", "cmp": ">", "value": 1},
                {"op": "select", "columns": ["id"]},
                {"op": "sort", "keys": [{"column": "id", "ascending": True}]},
            ],
        ),
    )
    unsafe = Case(
        "case-unsafe",
        1,
        case.tables,
        Program("prog-unsafe", 1, [{"op": "groupby", "keys": ["id"], "aggs": [{"func": "count", "column": "x", "as": "n"}]}]),
    )

    contract = infer_witness_contract(case)

    assert contract is not None
    assert contract.kind == "row_containment"
    assert contract.row == {"id": 2}
    assert contract.source == "inferred_row_preserving_pipeline"
    assert infer_witness_contract(unsafe) is None


def test_witness_predicate_rectification_reuses_filter_semantics():
    row = {"x": 3, "flag": None}

    assert row_satisfies_predicate(row, {"column": "x", "cmp": ">=", "value": 3}) is True
    assert row_satisfies_predicate(row, {"column": "flag", "cmp": "bool_is_unknown", "value": None}) is True

    rectified = rectify_predicate_for_row(row, {"column": "x", "cmp": ">", "value": 10})

    assert rectified["changed"] is True
    assert rectified["predicate"] == {"column": "x", "cmp": "==", "value": 3}
    assert row_satisfies_predicate(row, rectified["predicate"]) is True


def test_unsupported_witness_kind_is_reported_without_finding():
    case = _case_with_contract({"kind": "full_table_equivalence", "row": {"id": 1}})
    result = evaluate_witness_contract(
        case,
        {"left": NormalizedResult("left", "ok", ["id"], [[1]])},
        enabled=True,
    )

    payload = result.to_dict()
    assert payload["unsupported_reason"] == "unsupported_witness_kind:full_table_equivalence"
    assert finding_from_witness_result(case, result) is None
