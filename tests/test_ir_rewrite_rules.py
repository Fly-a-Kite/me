from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.mutator import IR_REWRITE_MUTATION_OPERATOR_NAMES, mutate_case_with_metadata
from datadiff.mutator_ir import (
    ir_rewrite_rule_for_operator,
    ir_rewrite_rule_registry_payload,
    metamorphic_rewrite_rule_metadata,
    registered_ir_rewrite_rules,
)


def test_ir_rewrite_rule_registry_covers_mutation_operators():
    rules = registered_ir_rewrite_rules()
    by_operator = {rule.operator: rule for rule in rules}

    assert set(IR_REWRITE_MUTATION_OPERATOR_NAMES) == set(by_operator)
    assert by_operator["ir_pushdown_filter"].semantics_class == "semantics_preserving"
    assert by_operator["ir_pull_filter_above_groupby"].semantics_class == "semantics_changing_probe"
    assert by_operator["ir_wrap_with_window"].semantics_class == "optimizer_boundary"
    assert "null" in by_operator["ir_pushdown_filter"].contract_axes


def test_ir_rewrite_registry_exposes_shared_mutation_metamorphic_reducer_payload():
    payload = ir_rewrite_rule_registry_payload()
    preserving = {row["operator"] for row in metamorphic_rewrite_rule_metadata()}

    assert payload["schema_version"] == "ir-rewrite-rule-registry-v1"
    assert "semantics_preserving" in payload["semantics_classes"]
    assert {"ir_swap_adjacent", "ir_pushdown_filter", "ir_fold_redundant_op"}.issubset(preserving)
    assert ir_rewrite_rule_for_operator("missing") is None
    assert all(row["reducer_role"] for row in payload["rules"])


def test_ir_rewrite_mutation_metadata_carries_rule_identity():
    left = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 1, "x": 10}, {"id": 2, "x": -1}],
    )
    right = TableData(
        "t1",
        [ColumnSpec("id", "int"), ColumnSpec("tag", "str")],
        [{"id": 1, "tag": "a"}, {"id": 2, "tag": "b"}],
    )
    base = Case(
        "case-ir-rule-metadata",
        10,
        [left, right],
        Program(
            "prog-ir-rule-metadata",
            10,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            ],
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_pushdown_filter": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )
    rule = result.metadata["mutation"]["ir_rewrite_rule"]

    assert rule["operator"] == "ir_pushdown_filter"
    assert rule["rule_id"] == "ir.rewrite.filter_pushdown"
    assert rule["semantics_class"] == "semantics_preserving"
    assert result.metadata["mutation"]["ir_rewrite_rules"] == [rule]
