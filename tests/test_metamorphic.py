from datadiff.datagen import generate_case
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.metamorphic import (
    all_metamorphic_variants,
    build_metamorphic_variants,
    build_metamorphic_variants_for_relations,
    evaluate_metamorphic_variants,
    ir_rewrite_metamorphic_rule_registry,
    select_metamorphic_variants,
)
from datadiff.normalizer import NormalizedResult
from datadiff.semantic_values import LOSSLESS_VALUE_SCHEMA_VERSION, encode_semantic_value


def test_metamorphic_builds_row_permutation_without_limit():
    case = generate_case(7)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "select", "columns": ["id"]}])
    variants = build_metamorphic_variants(case, limit=20)
    assert any(v.relation == "row_permutation" for v in variants)


def test_metamorphic_relation_order_prioritizes_requested_relation_under_limit():
    case = generate_case(7)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "select", "columns": ["id"]}])
    variants = all_metamorphic_variants(case)

    selected = select_metamorphic_variants(
        variants,
        limit=1,
        relation_order=["row_permutation"],
    )

    assert [variant.relation for variant in selected] == ["row_permutation"]


def test_metamorphic_exposes_semantics_preserving_ir_rewrite_rule_registry():
    registry = ir_rewrite_metamorphic_rule_registry()
    operators = {row["operator"] for row in registry["rules"]}

    assert registry["schema_version"] == "ir-rewrite-metamorphic-registry-v1"
    assert {"ir_swap_adjacent", "ir_pushdown_filter", "ir_fold_redundant_op"}.issubset(operators)
    assert "ir_pull_filter_above_groupby" not in operators
    assert registry["relations"]["filter_pushdown"]["operator"] == "ir_pushdown_filter"


def test_semi_anti_rewrite_compares_bag_semantics_not_physical_order():
    case = Case(
        "case-rewrite-order",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int")],
                [{"x": 1}, {"x": 2}],
            )
        ],
        Program(
            "prog-rewrite-order",
            1,
            [{"op": "sort", "columns": ["x"], "ascending": True}],
        ),
    )
    base = {"backend": NormalizedResult("backend", "ok", ["x"], [[1], [2]])}
    variants = {
        "semi_anti_join_rewrite:semi_join-op-0": {
            "backend": NormalizedResult(
                "backend",
                "ok",
                ["x"],
                [[2], [1]],
            )
        }
    }

    findings = evaluate_metamorphic_variants(case, base, variants)

    assert findings == []


def test_metamorphic_ignores_float_precision_only_with_explicit_tolerance():
    case = generate_case(9)
    case.metadata["semantic_comparison"] = {
        "view": "numeric_tolerant",
        "numeric_decimals": 10,
    }
    base = {"polars_lazy": NormalizedResult("polars_lazy", "ok", ["x"], [[0.11111111111111109]])}
    variants = {
        "join_inner_left_equivalence:0": {
            "polars_lazy": NormalizedResult("polars_lazy", "ok", ["x"], [[0.1111111111111111]])
        }
    }

    findings = evaluate_metamorphic_variants(case, base, variants)

    assert findings == []


def test_metamorphic_finding_records_status_mismatch_class():
    case = generate_case(10)
    base = {"duckdb": NormalizedResult("duckdb", "ok", ["x"], [[1]])}
    variants = {
        "filter_idempotence:duplicate-0": {
            "duckdb": NormalizedResult("duckdb", "error", [], [], "BinderException", "missing column")
        }
    }

    findings = evaluate_metamorphic_variants(case, base, variants)

    assert len(findings) == 1
    assert findings[0].mismatch_class == "status"
    assert "mismatch_class=status" in findings[0].evidence


def test_metamorphic_method_arm_can_replay_legacy_lossy_comparison():
    case = Case(
        "case-mr-method-comparison",
        1,
        [TableData("t0", [ColumnSpec("x", "float")], [{"x": float("nan")}])],
        Program("prog-mr-method-comparison", 1, [{"op": "select", "columns": ["x"]}]),
    )
    base_result = NormalizedResult(
        "engine",
        "ok",
        ["x"],
        [[None]],
        lossless_rows=[[encode_semantic_value(None)]],
        lossless_schema_version=LOSSLESS_VALUE_SCHEMA_VERSION,
    )
    variant_result = NormalizedResult(
        "engine",
        "ok",
        ["x"],
        [[None]],
        lossless_rows=[[encode_semantic_value(float("nan"))]],
        lossless_schema_version=LOSSLESS_VALUE_SCHEMA_VERSION,
    )
    variants = {"filter_idempotence:duplicate-0": {"engine": variant_result}}

    assert evaluate_metamorphic_variants(
        case,
        {"engine": base_result},
        variants,
        comparison_mode="legacy",
    ) == []
    findings = evaluate_metamorphic_variants(
        case,
        {"engine": base_result},
        variants,
        comparison_mode="contract",
    )
    assert len(findings) == 1
    assert "contract" not in findings[0].oracle
    assert "view=" in findings[0].evidence


def test_metamorphic_skips_row_permutation_with_limit():
    case = generate_case(8)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "limit", "n": 1}])
    variants = build_metamorphic_variants(case, limit=20)
    assert not any(v.relation == "row_permutation" for v in variants)


def test_row_permutation_preserves_join_tables():
    case = Case(
        "case-join",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 1, "j": 100}, {"id": 2, "j": 200}],
            ),
        ],
        Program(
            "prog-join",
            1,
            [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"}],
        ),
    )

    variant = next(v for v in build_metamorphic_variants(case, limit=20) if v.relation == "row_permutation")

    assert [table.name for table in variant.case.tables] == ["t0", "t1"]
    assert variant.case.tables[0].rows == [{"id": 2, "x": 20}, {"id": 1, "x": 10}]
    assert variant.case.tables[1].rows == case.tables[1].rows


def test_metamorphic_builds_filter_idempotence():
    case = generate_case(9)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "filter_idempotence")
    assert variant.case.program.operations == [
        {"op": "filter", "column": "id", "cmp": ">=", "value": 0},
        {"op": "filter", "column": "id", "cmp": ">=", "value": 0},
    ]


def test_metamorphic_builds_distinct_idempotence():
    case = generate_case(9)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [{"op": "distinct", "columns": ["g", "x"]}],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "distinct_idempotence")
    assert variant.case.program.operations == [
        {"op": "distinct", "columns": ["g", "x"]},
        {"op": "distinct", "columns": ["g", "x"]},
    ]


def test_metamorphic_builds_fill_null_idempotence():
    case = generate_case(9)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [{"op": "fill_null", "column": "x", "value": 0}],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "fill_null_idempotence")
    assert variant.case.program.operations == [
        {"op": "fill_null", "column": "x", "value": 0},
        {"op": "fill_null", "column": "x", "value": 0},
    ]


def test_metamorphic_builds_coalesce_idempotence():
    case = generate_case(9)
    op = {"op": "coalesce", "columns": ["g", "s"], "as": "g_or_s", "fallback": "missing"}
    case.program = Program(case.program.program_id, case.program.seed, [op])

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "coalesce_idempotence")
    assert variant.case.program.operations == [op, op]


def test_metamorphic_builds_case_when_idempotence():
    case = generate_case(9)
    op = {
        "op": "case_when",
        "as": "label",
        "condition": {"column": "x", "cmp": ">=", "value": 0},
        "then": "yes",
        "else": "no",
    }
    case.program = Program(case.program.program_id, case.program.seed, [op])

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "case_when_idempotence")
    assert variant.case.program.operations == [op, op]


def test_metamorphic_builds_union_all_empty_append():
    case = generate_case(9)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}])

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "union_all_empty_append")
    assert variant.case.tables[-1].name == "t_empty_union"
    assert variant.case.tables[-1].rows == []
    assert variant.case.program.operations == [
        {"op": "union_all", "table": "t_empty_union"},
        {"op": "filter", "column": "id", "cmp": ">=", "value": 0},
    ]


def test_metamorphic_builds_filter_input_materialization_boundary():
    case = Case(
        "case-filter-materialization",
        11,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str"), ColumnSpec("x", "int")],
                [
                    {"id": 1, "s": "Alpha", "x": 10},
                    {"id": 2, "s": None, "x": 20},
                    {"id": 3, "s": "Beta", "x": 30},
                    {"id": 4, "s": "Gamma", "x": 40},
                ],
            )
        ],
        Program(
            "prog-filter-materialization",
            11,
            [
                {"op": "filter", "column": "s", "cmp": "in_set", "value": ["Alpha", "Beta"]},
                {
                    "op": "groupby",
                    "keys": ["s"],
                    "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                },
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=1)

    assert len(variants) == 1
    variant = variants[0]
    assert variant.name == "filter_input_materialization:s-4to2"
    assert variant.relation == "filter_input_materialization"
    assert variant.case.tables[0].rows == [
        {"id": 1, "s": "Alpha", "x": 10},
        {"id": 3, "s": "Beta", "x": 30},
    ]
    assert variant.case.program.operations == [
        {
            "op": "groupby",
            "keys": ["s"],
            "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
        }
    ]


def test_metamorphic_skips_filter_input_materialization_for_nonleading_filters():
    case = generate_case(12)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [
            {"op": "mutate", "column": "x0", "expr": {"kind": "add_const", "source": "x", "value": 0}},
            {"op": "filter", "column": "id", "cmp": ">=", "value": 1},
        ],
    )

    variants = build_metamorphic_variants(case, limit=40)

    assert not any(v.relation == "filter_input_materialization" for v in variants)


def test_metamorphic_builds_drop_nulls_input_materialization_boundary():
    case = Case(
        "case-drop-nulls-materialization",
        13,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 2, "g": None, "x": 20},
                    {"id": 3, "g": "b", "x": None},
                    {"id": 4, "g": "b", "x": 40},
                ],
            )
        ],
        Program(
            "prog-drop-nulls-materialization",
            13,
            [
                {"op": "drop_nulls", "columns": ["g", "x"]},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                },
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=1)

    variant = variants[0]
    assert variant.name == "drop_nulls_input_materialization:g-x-4to2"
    assert variant.relation == "drop_nulls_input_materialization"
    assert variant.case.tables[0].rows == [
        {"id": 1, "g": "a", "x": 10},
        {"id": 4, "g": "b", "x": 40},
    ]
    assert [op["op"] for op in variant.case.program.operations] == ["groupby"]


def test_metamorphic_builds_fill_null_input_materialization_boundary():
    case = Case(
        "case-fill-null-materialization",
        14,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 2, "g": None, "x": 20},
                    {"id": 3, "g": "b", "x": None},
                ],
            )
        ],
        Program(
            "prog-fill-null-materialization",
            14,
            [
                {"op": "fill_null", "column": "g", "value": "missing"},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "id", "func": "count", "as": "count_id"}],
                },
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=1)

    variant = variants[0]
    assert variant.name == "fill_null_input_materialization:g"
    assert variant.relation == "fill_null_input_materialization"
    assert variant.case.tables[0].rows == [
        {"id": 1, "g": "a", "x": 10},
        {"id": 2, "g": "missing", "x": 20},
        {"id": 3, "g": "b", "x": None},
    ]
    assert [op["op"] for op in variant.case.program.operations] == ["groupby"]


def test_metamorphic_builds_distinct_input_materialization_boundary():
    case = Case(
        "case-distinct-materialization",
        15,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 2, "g": "a", "x": 10},
                    {"id": 3, "g": None, "x": 20},
                    {"id": 4, "g": None, "x": 20},
                ],
            )
        ],
        Program(
            "prog-distinct-materialization",
            15,
            [
                {"op": "distinct", "columns": ["g", "x"]},
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                },
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=1)

    variant = variants[0]
    assert variant.name == "distinct_input_materialization:g-x-4to2"
    assert variant.relation == "distinct_input_materialization"
    assert [column.name for column in variant.case.tables[0].columns] == ["g", "x"]
    assert variant.case.tables[0].rows == [
        {"g": "a", "x": 10},
        {"g": None, "x": 20},
    ]
    assert [op["op"] for op in variant.case.program.operations] == ["groupby"]


def test_metamorphic_builds_input_partition_union_all_boundary():
    case = Case(
        "case-partition",
        12,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 2, "g": "b", "x": 20},
                    {"id": 3, "g": "a", "x": 30},
                    {"id": 4, "g": "b", "x": 40},
                ],
            )
        ],
        Program(
            "prog-partition",
            12,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
                }
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=1)

    variant = variants[0]
    assert variant.relation == "input_partition_union_all"
    assert variant.name == "input_partition_union_all:split-2-2"
    assert [table.name for table in variant.case.tables] == ["t0", "t_partition_tail"]
    assert variant.case.tables[0].rows == [
        {"id": 1, "g": "a", "x": 10},
        {"id": 2, "g": "b", "x": 20},
    ]
    assert variant.case.tables[1].rows == [
        {"id": 3, "g": "a", "x": 30},
        {"id": 4, "g": "b", "x": 40},
    ]
    assert variant.case.program.operations == [
        {"op": "union_all", "table": "t_partition_tail"},
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}],
        },
    ]


def test_metamorphic_skips_input_partition_union_all_for_order_sensitive_programs():
    case = generate_case(12)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "limit", "n": 2}])

    variants = build_metamorphic_variants(case, limit=40)

    assert not any(v.relation == "input_partition_union_all" for v in variants)


def test_metamorphic_builds_drop_nulls_idempotence():
    case = generate_case(9)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "drop_nulls", "columns": ["g", "x"]}])

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "drop_nulls_idempotence")
    assert variant.case.program.operations == [
        {"op": "drop_nulls", "columns": ["g", "x"]},
        {"op": "drop_nulls", "columns": ["g", "x"]},
    ]


def test_metamorphic_builds_semi_anti_join_right_duplicate_injection():
    case = Case(
        "case-semi-anti",
        11,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": 10}]),
            TableData(
                "t_lookup",
                [ColumnSpec("id", "int"), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": "one"}, {"id": None, "tag": "null-key"}],
            ),
        ],
        Program("prog-semi-anti", 11, [{"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}]),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "semi_anti_join_right_duplicate_injection")
    assert variant.case.tables[1].rows == [
        {"id": 1, "tag": "one"},
        {"id": None, "tag": "null-key"},
        {"id": 1, "tag": "one"},
    ]
    assert variant.case.program.operations == [
        {"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}
    ]


def test_metamorphic_builds_multi_key_semi_anti_join_right_duplicate_injection():
    case = Case(
        "case-multi-key-semi-anti",
        11,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"id": 1, "g": "a", "x": 10}],
            ),
            TableData(
                "t_lookup",
                [ColumnSpec("rid", "int"), ColumnSpec("rg", "str"), ColumnSpec("tag", "str")],
                [
                    {"rid": 1, "rg": "a", "tag": "one"},
                    {"rid": None, "rg": "a", "tag": "null-key"},
                ],
            ),
        ],
        Program(
            "prog-multi-key-semi-anti",
            11,
            [{"op": "semi_join", "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["rid", "rg"]}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "semi_anti_join_right_duplicate_injection")
    assert variant.case.tables[1].rows[-1] == {"rid": 1, "rg": "a", "tag": "one"}
    assert variant.case.program.operations == case.program.operations


def test_metamorphic_skips_right_duplicate_when_table_is_reused_elsewhere():
    case = Case(
        "case-reused-semi-right",
        11,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int")],
                [{"id": 1}],
            ),
            TableData(
                "lookup",
                [ColumnSpec("id", "int")],
                [{"id": 1}],
            ),
        ],
        Program(
            "prog-reused-semi-right",
            11,
            [
                {
                    "op": "join",
                    "table": "lookup",
                    "left_on": "id",
                    "right_on": "id",
                    "how": "inner",
                },
                {
                    "op": "semi_join",
                    "table": "lookup",
                    "left_on": "id",
                    "right_on": "id",
                },
            ],
        ),
    )

    variants = build_metamorphic_variants_for_relations(
        case,
        ["semi_anti_join_right_duplicate_injection"],
    )

    assert variants == []


def test_metamorphic_builds_semi_anti_join_unmatched_right_injection():
    case = Case(
        "case-semi-anti-unmatched",
        11,
        [
            TableData("t0", [ColumnSpec("id", "int"), ColumnSpec("x", "int")], [{"id": 1, "x": 10}]),
            TableData(
                "t_lookup",
                [ColumnSpec("id", "int"), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": "one"}],
            ),
        ],
        Program("prog-semi-anti-unmatched", 11, [{"op": "anti_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}]),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "semi_anti_join_unmatched_right_injection")
    assert variant.case.tables[1].rows[-1]["id"] not in {1}
    assert variant.case.tables[1].rows[-1]["tag"] == ""
    assert variant.case.program.operations == case.program.operations


def test_metamorphic_builds_multi_key_semi_anti_join_unmatched_right_injection():
    case = Case(
        "case-multi-key-semi-anti-unmatched",
        11,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"id": 1, "g": "a", "x": 10}, {"id": 2, "g": "b", "x": 20}],
            ),
            TableData(
                "t_lookup",
                [ColumnSpec("rid", "int"), ColumnSpec("rg", "str"), ColumnSpec("tag", "str")],
                [{"rid": 1, "rg": "a", "tag": "one"}],
            ),
        ],
        Program(
            "prog-multi-key-semi-anti-unmatched",
            11,
            [{"op": "semi_join", "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["rid", "rg"]}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "semi_anti_join_unmatched_right_injection")
    injected = variant.case.tables[1].rows[-1]
    assert (injected["rid"], injected["rg"]) not in {(1, "a"), (2, "b")}
    assert injected["tag"] == ""
    assert variant.case.program.operations == case.program.operations


def test_metamorphic_rewrites_semi_join_to_distinct_inner_join():
    case = Case(
        "case-semi-join-rewrite",
        12,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}, {"id": None, "x": 30}],
            ),
            TableData(
                "t_lookup",
                [ColumnSpec("rid", "int"), ColumnSpec("tag", "str")],
                [
                    {"rid": 1, "tag": "one"},
                    {"rid": 1, "tag": "duplicate"},
                    {"rid": None, "tag": "null-key"},
                    {"rid": 3, "tag": "three"},
                ],
            ),
        ],
        Program(
            "prog-semi-join-rewrite",
            12,
            [{"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "rid"}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "semi_anti_join_rewrite")
    key_table = variant.case.tables[-1]
    assert key_table.name == "t_mr_semi_join_keys_0"
    assert [column.name for column in key_table.columns] == ["rid"]
    assert key_table.rows == [{"rid": 1}, {"rid": 3}]
    assert variant.case.program.operations == [
        {
            "op": "join",
            "table": "t_mr_semi_join_keys_0",
            "left_on": ["id"],
            "right_on": ["rid"],
            "how": "inner",
        }
    ]


def test_metamorphic_rewrites_multi_key_anti_join_to_left_join_marker_filter():
    case = Case(
        "case-anti-join-rewrite",
        13,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"id": 1, "g": "a", "x": 10}, {"id": 2, "g": "b", "x": 20}],
            ),
            TableData(
                "t_lookup",
                [ColumnSpec("rid", "int"), ColumnSpec("rg", "str"), ColumnSpec("tag", "str")],
                [
                    {"rid": 1, "rg": "a", "tag": "one"},
                    {"rid": 1, "rg": "a", "tag": "duplicate"},
                    {"rid": 2, "rg": None, "tag": "null-key"},
                ],
            ),
        ],
        Program(
            "prog-anti-join-rewrite",
            13,
            [
                {"op": "mutate", "column": "x0", "expr": {"kind": "add_const", "source": "x", "value": 0}},
                {"op": "anti_join", "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["rid", "rg"]},
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x0", "func": "sum", "as": "sum_x0"}]},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "semi_anti_join_rewrite")
    key_table = variant.case.tables[-1]
    assert [column.name for column in key_table.columns] == ["rid", "rg", "__datadiff_mr_match"]
    assert key_table.rows == [{"rid": 1, "rg": "a", "__datadiff_mr_match": 1}]
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "x0", "expr": {"kind": "add_const", "source": "x", "value": 0}},
        {
            "op": "join",
            "table": "t_mr_anti_join_keys_1",
            "left_on": ["id", "g"],
            "right_on": ["rid", "rg"],
            "how": "left",
        },
        {"op": "filter", "column": "__datadiff_mr_match", "cmp": "is_null", "value": None},
        {"op": "select", "columns": ["id", "g", "x", "x0"]},
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x0", "func": "sum", "as": "sum_x0"}]},
    ]


def test_metamorphic_builds_limit_idempotence():
    case = generate_case(10)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "limit", "n": 3}])

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "limit_idempotence")
    assert variant.case.program.operations == [{"op": "limit", "n": 3}, {"op": "limit", "n": 3}]


def test_metamorphic_builds_offset_zero_insertion():
    case = generate_case(10)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [{"op": "sort", "columns": ["id"], "ascending": True}, {"op": "select", "columns": ["id", "g"]}],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "offset_zero_insertion")
    assert variant.case.program.operations == [
        {"op": "sort", "columns": ["id"], "ascending": True},
        {"op": "select", "columns": ["id", "g"]},
        {"op": "offset", "n": 0},
    ]


def test_metamorphic_builds_offset_limit_fusion():
    case = generate_case(10)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [
            {"op": "sort", "columns": ["id"], "ascending": True},
            {"op": "offset", "n": 2},
            {"op": "limit", "n": 3},
        ],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "offset_limit_fusion")
    assert variant.case.program.operations == [
        {"op": "sort", "columns": ["id"], "ascending": True},
        {"op": "limit", "n": 5},
        {"op": "offset", "n": 2},
    ]


def test_metamorphic_builds_limit_offset_fusion():
    case = generate_case(10)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [
            {"op": "sort", "columns": ["id"], "ascending": True},
            {"op": "limit", "n": 5},
            {"op": "offset", "n": 2},
        ],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "limit_offset_fusion")
    assert variant.case.program.operations == [
        {"op": "sort", "columns": ["id"], "ascending": True},
        {"op": "offset", "n": 2},
        {"op": "limit", "n": 3},
    ]


def test_metamorphic_builds_sort_idempotence():
    case = generate_case(12)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [{"op": "sort", "columns": ["id", "g"], "ascending": False}],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "sort_idempotence")
    assert variant.case.program.operations == [
        {"op": "sort", "columns": ["id", "g"], "ascending": False},
        {"op": "sort", "columns": ["id", "g"], "ascending": False},
    ]


def test_metamorphic_builds_groupby_key_permutation():
    case = generate_case(11, profile="workflow")
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [
            {
                "op": "groupby",
                "keys": ["flag", "g"],
                "aggs": [{"column": "x", "func": "count", "as": "count_x"}],
            }
        ],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "groupby_key_permutation")
    assert variant.case.program.operations[0]["keys"] == ["g", "flag"]


def test_metamorphic_builds_groupby_aggregation_permutation():
    case = generate_case(15, profile="workflow")
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "count", "as": "count_x"},
                    {"column": "id", "func": "sum", "as": "sum_id"},
                ],
            }
        ],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "groupby_aggregation_permutation")
    assert variant.case.program.operations[0]["aggs"] == [
        {"column": "id", "func": "sum", "as": "sum_id"},
        {"column": "x", "func": "count", "as": "count_x"},
    ]


def test_metamorphic_builds_groupby_sorted_input_for_exact_aggregates():
    case = Case(
        "case-groupby-sorted-input",
        16,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("g", "str"),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("x", "int"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"g": "b", "flag": True, "x": 2, "s": "beta"},
                    {"g": "a", "flag": None, "x": 1, "s": "alpha"},
                    {"g": "a", "flag": False, "x": 1, "s": None},
                ],
            )
        ],
        Program(
            "prog-groupby-sorted-input",
            16,
            [
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [
                        {"column": "x", "func": "nunique", "as": "nunique_x"},
                        {"column": "flag", "func": "any", "as": "any_flag"},
                    ],
                }
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "groupby_sorted_input")
    assert variant.case.program.operations == [
        {
            "op": "sort",
            "keys": [
                {"column": "g", "ascending": True, "nulls": "last"},
                {"column": "flag", "ascending": False, "nulls": "first"},
                {"column": "x", "ascending": True, "nulls": "last"},
            ],
        },
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [
                {"column": "x", "func": "nunique", "as": "nunique_x"},
                {"column": "flag", "func": "any", "as": "any_flag"},
            ],
        },
    ]


def test_metamorphic_skips_groupby_sorted_input_for_mean_aggregates():
    case = Case(
        "case-groupby-sorted-input-mean",
        161,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "float")],
                [{"g": "a", "x": 0.1}, {"g": "a", "x": 0.2}],
            )
        ],
        Program(
            "prog-groupby-sorted-input-mean",
            161,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "mean", "as": "mean_x"}]}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(v.relation == "groupby_sorted_input" for v in variants)


def test_metamorphic_skips_groupby_sorted_input_before_order_observing_tail():
    case = Case(
        "case-groupby-sorted-input-limit-tail",
        162,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"g": "b", "x": 2}, {"g": "a", "x": 1}],
            )
        ],
        Program(
            "prog-groupby-sorted-input-limit-tail",
            162,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(v.relation == "groupby_sorted_input" for v in variants)


def test_metamorphic_skips_groupby_sorted_input_when_tail_sorts_before_limit():
    case = Case(
        "case-groupby-sorted-input-sort-limit-tail",
        163,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"g": "b", "x": 2}, {"g": "a", "x": 1}],
            )
        ],
        Program(
            "prog-groupby-sorted-input-sort-limit-tail",
            163,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "count", "as": "count_x"}]},
                {"op": "sort", "keys": [{"column": "g", "ascending": True, "nulls": "last"}]},
                {"op": "limit", "n": 1},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(v.relation == "groupby_sorted_input" for v in variants)


def test_metamorphic_groupby_sorted_input_tracks_columns_after_global_aggregate():
    case = Case(
        "case-groupby-sorted-input-after-aggregate",
        164,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int"),
                    ColumnSpec("g", "str"),
                    ColumnSpec("flag", "bool"),
                    ColumnSpec("s", "str"),
                ],
                [
                    {"id": 1, "g": "a", "flag": True, "s": "alpha"},
                    {"id": 2, "g": "b", "flag": False, "s": "beta"},
                ],
            )
        ],
        Program(
            "prog-groupby-sorted-input-after-aggregate",
            164,
            [
                {
                    "op": "aggregate",
                    "aggs": [
                        {"column": "g", "func": "nunique", "as": "unique_g"},
                        {"column": "s", "func": "nunique", "as": "unique_s"},
                        {"column": "flag", "func": "nunique", "as": "unique_flag"},
                    ],
                },
                {
                    "op": "groupby",
                    "keys": ["unique_s"],
                    "aggs": [{"column": "unique_flag", "func": "nunique", "as": "nunique_unique_flag"}],
                },
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "groupby_sorted_input")
    sort_columns = [key["column"] for key in variant.case.program.operations[1]["keys"]]
    assert sort_columns == ["unique_s", "unique_g", "unique_flag"]


def test_metamorphic_builds_groupby_neutral_mutation():
    case = Case(
        "case-groupby-neutral",
        17,
        [
            TableData(
                "t0",
                [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
                [{"g": "a", "x": 1}, {"g": "a", "x": None}],
            )
        ],
        Program(
            "prog-groupby-neutral",
            17,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "groupby_neutral_mutation")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "mr_x_plus0", "expr": {"kind": "add_const", "source": "x", "value": 0}},
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "mr_x_plus0", "func": "sum", "as": "sum_x"}]},
    ]


def test_metamorphic_skips_groupby_neutral_mutation_for_boolean_aggregate_outputs():
    case = Case(
        "case-groupby-neutral-bool",
        171,
        [TableData("t0", [ColumnSpec("g", "str"), ColumnSpec("flag", "bool")], [{"g": None, "flag": True}])],
        Program(
            "prog-groupby-neutral-bool",
            171,
            [
                {
                    "op": "case_when",
                    "as": "flag_bucket",
                    "condition": {"column": "flag", "cmp": "bool_is_true", "value": None},
                    "then": "true_member",
                    "else": "false_member",
                },
                {
                    "op": "groupby",
                    "keys": ["g"],
                    "aggs": [{"column": "flag", "func": "any", "as": "any_flag"}],
                },
                {
                    "op": "groupby",
                    "keys": ["flag_bucket"],
                    "aggs": [{"column": "any_flag", "func": "count", "as": "count_rows"}],
                },
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=40)

    assert not any(v.relation == "groupby_neutral_mutation" for v in variants)


def test_metamorphic_builds_mutate_add_zero_insertion():
    case = Case(
        "case-mutate-zero",
        18,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 2}],
            )
        ],
        Program("prog-mutate-zero", 18, [{"op": "select", "columns": ["id", "x"]}]),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "mutate_add_zero_insertion")
    assert variant.case.program.operations[0] == {
        "op": "mutate",
        "column": "id",
        "expr": {"kind": "add_const", "source": "id", "value": 0},
    }


def test_metamorphic_builds_join_inner_left_equivalence_when_keys_are_covered():
    case = Case(
        "case-join-covered",
        19,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 1, "j": 100}, {"id": 2, "j": 200}, {"id": 2, "j": 201}],
            ),
        ],
        Program(
            "prog-join-covered",
            19,
            [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "join_inner_left_equivalence")
    assert variant.case.program.operations[0]["how"] == "inner"


def test_metamorphic_skips_join_inner_left_equivalence_for_null_left_keys():
    case = Case(
        "case-join-null-left-key",
        1901,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=True), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": None, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=True), ColumnSpec("j", "int")],
                [{"id": 1, "j": 100}, {"id": None, "j": 200}],
            ),
        ],
        Program(
            "prog-join-null-left-key",
            1901,
            [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(variant.relation == "join_inner_left_equivalence" for variant in variants)


def test_metamorphic_builds_join_filter_pushdown_for_left_column():
    case = Case(
        "case-join-filter-pushdown",
        20,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 1, "j": 100}, {"id": 2, "j": 200}],
            ),
        ],
        Program(
            "prog-join-filter-pushdown",
            20,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "filter", "column": "x", "cmp": ">=", "value": 10},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "join_filter_pushdown")
    assert [op["op"] for op in variant.case.program.operations] == ["filter", "join"]


def test_metamorphic_builds_filter_mutate_commutation_for_independent_columns():
    case = Case(
        "case-filter-mutate-commute",
        21,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            )
        ],
        Program(
            "prog-filter-mutate-commute",
            21,
            [
                {"op": "filter", "column": "id", "cmp": ">=", "value": 1},
                {"op": "mutate", "column": "x2", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "filter_mutate_commutation")
    assert [op["op"] for op in variant.case.program.operations] == ["mutate", "filter"]


def test_metamorphic_builds_filter_tautology_insertion():
    case = generate_case(16)
    case.program = Program(case.program.program_id, case.program.seed, [{"op": "select", "columns": ["id", "x"]}])

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "filter_tautology_insertion")
    assert variant.case.program.operations[0] == {"op": "filter", "column": "id", "cmp": ">=", "value": 0}
    assert variant.case.program.operations[1:] == [{"op": "select", "columns": ["id", "x"]}]


def test_metamorphic_skips_filter_tautology_insertion_for_nullable_id():
    case = Case(
        "case-null-id-tautology",
        22,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=True), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": None, "x": 20}],
            )
        ],
        Program("prog-null-id-tautology", 22, [{"op": "select", "columns": ["id", "x"]}]),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(v.relation == "filter_tautology_insertion" for v in variants)


def test_metamorphic_skips_filter_tautology_insertion_for_mean_sort_boundary():
    case = Case(
        "case-mean-sort-tautology",
        23,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("g", "str"), ColumnSpec("x", "float")],
                [{"id": 1, "g": "a", "x": 0.1}, {"id": 2, "g": "a", "x": 0.2}],
            )
        ],
        Program(
            "prog-mean-sort-tautology",
            23,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "mean", "as": "mean_x"}]},
                {"op": "sort", "keys": [{"column": "mean_x", "ascending": True, "nulls": "last"}]},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(v.relation == "filter_tautology_insertion" for v in variants)


def test_program_order_sensitive_when_row_number_precedes_groupby():
    case = Case(
        "case-row-number-groupby",
        24,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [{"id": 1, "g": "a", "x": 10}, {"id": 2, "g": "a", "x": 20}],
            )
        ],
        Program(
            "prog-row-number-groupby",
            24,
            [
                {
                    "op": "row_number_filter",
                    "partition_by": ["g"],
                    "order_by": [{"column": "x", "ascending": True, "nulls": "last"}],
                    "cmp": "==",
                    "value": 1,
                },
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
            ],
        ),
    )

    assert case.program.order_sensitive


def test_metamorphic_skips_overtrigger_relations_with_row_number_observer():
    case = Case(
        "case-row-number-overtrigger",
        25,
        [
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("g", "str"),
                    ColumnSpec("x", "int"),
                ],
                [
                    {"id": 1, "g": "a", "x": 10},
                    {"id": 2, "g": "a", "x": 20},
                    {"id": 3, "g": "b", "x": 30},
                    {"id": 4, "g": "b", "x": 40},
                ],
            )
        ],
        Program(
            "prog-row-number-overtrigger",
            25,
            [
                {
                    "op": "row_number_filter",
                    "partition_by": ["g"],
                    "order_by": [{"column": "x", "ascending": True, "nulls": "last"}],
                    "cmp": "==",
                    "value": 1,
                },
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
            ],
        ),
    )

    relations = {variant.relation for variant in build_metamorphic_variants(case, limit=100)}

    assert "input_partition_union_all" not in relations
    assert "filter_tautology_insertion" not in relations
    assert "groupby_neutral_mutation" not in relations
    assert "groupby_sorted_input" not in relations
    assert "groupby_aggregation_permutation" not in relations
    assert "mutate_add_zero_insertion" not in relations
    assert "row_permutation" not in relations


def test_metamorphic_skips_join_inner_left_equivalence_for_groupby_mean_tail():
    case = Case(
        "case-join-mean-tail",
        26,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "float")],
                [{"id": 1, "x": 0.1}, {"id": 2, "x": 0.2}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "float")],
                [{"id": 1, "j": 1.0}, {"id": 2, "j": 2.0}],
            ),
        ],
        Program(
            "prog-join-mean-tail",
            26,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "groupby", "keys": ["id"], "aggs": [{"column": "j", "func": "mean", "as": "mean_j"}]},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=100)

    assert not any(variant.relation == "join_inner_left_equivalence" for variant in variants)


def test_metamorphic_builds_sort_select_commutation():
    case = generate_case(13)
    case.program = Program(
        case.program.program_id,
        case.program.seed,
        [
            {"op": "select", "columns": ["g", "id", "x"]},
            {"op": "sort", "columns": ["id", "g"], "ascending": True},
        ],
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "sort_select_commutation")
    assert variant.case.program.operations == [
        {"op": "sort", "columns": ["id", "g"], "ascending": True},
        {"op": "select", "columns": ["g", "id", "x"]},
    ]


def test_join_table_permutation_reverses_secondary_table():
    case = Case(
        "case-join-secondary",
        2,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("j", "int")],
                [{"id": 1, "j": 100}, {"id": 2, "j": 200}],
            ),
        ],
        Program(
            "prog-join-secondary",
            2,
            [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "join_table_permutation")
    assert variant.case.tables[0].rows == case.tables[0].rows
    assert variant.case.tables[1].rows == [{"id": 2, "j": 200}, {"id": 1, "j": 100}]


def test_domain_metamorphic_injects_filter_rejecting_row():
    case = Case(
        "case-cleaning",
        3,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("flag", "bool")],
                [{"id": 1, "flag": True}, {"id": 2, "flag": True}],
            )
        ],
        Program("prog-cleaning", 3, [{"op": "filter", "column": "flag", "cmp": "==", "value": True}]),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "filter_rejecting_row_injection")
    assert variant.case.tables[0].rows[-1] == {"id": 0, "flag": False}
    assert variant.case.program.operations == case.program.operations


def test_domain_metamorphic_injects_unmatched_dimension_row():
    case = Case(
        "case-enrichment",
        4,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": "gold"}, {"id": 2, "tag": "silver"}],
            ),
        ],
        Program(
            "prog-enrichment",
            4,
            [{"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "join_unmatched_dimension_injection")
    assert variant.case.tables[0].rows == case.tables[0].rows
    assert variant.case.tables[1].rows[-1]["id"] not in {1, 2}
    assert variant.case.tables[1].rows[-1]["tag"] == ""


def test_domain_metamorphic_skips_unmatched_dimension_row_when_right_table_is_reused():
    case = Case(
        "case-enrichment-reused-right",
        1902,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("x", "int")],
                [{"id": 1, "x": 10}, {"id": 2, "x": 20}],
            ),
            TableData(
                "t1",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
                [{"id": 1, "tag": "gold"}, {"id": 2, "tag": "silver"}],
            ),
        ],
        Program(
            "prog-enrichment-reused-right",
            1902,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
                {"op": "semi_join", "table": "t1", "left_on": "id", "right_on": "id"},
            ],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    assert not any(variant.relation == "join_unmatched_dimension_injection" for variant in variants)


def test_domain_metamorphic_repeats_string_lower_normalization():
    case = Case(
        "case-log-normalization",
        5,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": "ERROR"}, {"id": 2, "s": "warn"}],
            )
        ],
        Program(
            "prog-log-normalization",
            5,
            [{"op": "mutate", "column": "level", "expr": {"kind": "string_lower", "source": "s"}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_lower_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "level", "expr": {"kind": "string_lower", "source": "s"}},
        {"op": "mutate", "column": "level", "expr": {"kind": "string_lower", "source": "level"}},
    ]


def test_domain_metamorphic_repeats_string_upper_normalization():
    case = Case(
        "case-upper-normalization",
        15,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": "error"}, {"id": 2, "s": "Warn"}],
            )
        ],
        Program(
            "prog-upper-normalization",
            15,
            [{"op": "mutate", "column": "level", "expr": {"kind": "string_upper", "source": "s"}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_upper_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "level", "expr": {"kind": "string_upper", "source": "s"}},
        {"op": "mutate", "column": "level", "expr": {"kind": "string_upper", "source": "level"}},
    ]


def test_domain_metamorphic_repeats_string_strip_normalization():
    case = Case(
        "case-strip-normalization",
        6,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": " ERROR "}, {"id": 2, "s": "warn"}],
            )
        ],
        Program(
            "prog-strip-normalization",
            6,
            [{"op": "mutate", "column": "level", "expr": {"kind": "string_strip", "source": "s"}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_strip_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "level", "expr": {"kind": "string_strip", "source": "s"}},
        {"op": "mutate", "column": "level", "expr": {"kind": "string_strip", "source": "level"}},
    ]


def test_domain_metamorphic_repeats_safe_string_replace_normalization():
    case = Case(
        "case-replace-normalization",
        7,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": "space value"}, {"id": 2, "s": "plain"}],
            )
        ],
        Program(
            "prog-replace-normalization",
            7,
            [{"op": "mutate", "column": "token", "expr": {"kind": "string_replace", "source": "s", "old": " ", "new": "_"}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_replace_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "token", "expr": {"kind": "string_replace", "source": "s", "old": " ", "new": "_"}},
        {"op": "mutate", "column": "token", "expr": {"kind": "string_replace", "source": "token", "old": " ", "new": "_"}},
    ]


def test_domain_metamorphic_repeats_string_slice_prefix_normalization():
    case = Case(
        "case-slice-normalization",
        8,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": "Alpha"}, {"id": 2, "s": "Beta"}],
            )
        ],
        Program(
            "prog-slice-normalization",
            8,
            [{"op": "mutate", "column": "prefix", "expr": {"kind": "string_slice", "source": "s", "start": 0, "length": 3}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_slice_prefix_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "prefix", "expr": {"kind": "string_slice", "source": "s", "start": 0, "length": 3}},
        {"op": "mutate", "column": "prefix", "expr": {"kind": "string_slice", "source": "prefix", "start": 0, "length": 3}},
    ]


def test_domain_metamorphic_repeats_string_null_if_empty_normalization():
    case = Case(
        "case-null-if-empty",
        9,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": ""}, {"id": 2, "s": "value"}],
            )
        ],
        Program(
            "prog-null-if-empty",
            9,
            [{"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_null_if_empty_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s"}},
        {"op": "mutate", "column": "s_norm", "expr": {"kind": "string_null_if_empty", "source": "s_norm"}},
    ]


def test_domain_metamorphic_repeats_string_split_part_first_token():
    case = Case(
        "case-split-first-token",
        9,
        [
            TableData(
                "t0",
                [ColumnSpec("id", "int", nullable=False), ColumnSpec("s", "str")],
                [{"id": 1, "s": "space value"}, {"id": 2, "s": "plain"}],
            )
        ],
        Program(
            "prog-split-first-token",
            9,
            [{"op": "mutate", "column": "token", "expr": {"kind": "string_split_part", "source": "s", "sep": " ", "index": 0}}],
        ),
    )

    variants = build_metamorphic_variants(case, limit=20)

    variant = next(v for v in variants if v.relation == "string_split_part_idempotence")
    assert variant.case.program.operations == [
        {"op": "mutate", "column": "token", "expr": {"kind": "string_split_part", "source": "s", "sep": " ", "index": 0}},
        {"op": "mutate", "column": "token", "expr": {"kind": "string_split_part", "source": "token", "sep": " ", "index": 0}},
    ]
