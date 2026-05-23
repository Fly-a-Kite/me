from datadiff.datagen import generate_case, repair_operations
from datadiff.classification_oracle import validate_case_program
from datadiff.dsl import sort_columns
from datadiff.guidance import extract_case_features
from datadiff.identifiers import is_reserved_output_name, make_safe_output_name


def _assert_program_columns_are_valid(case):
    table_by_name = {table.name: table for table in case.tables}
    known_cols = {c.name for c in case.tables[0].columns}
    for op in case.program.operations:
        if op["op"] == "join":
            right = table_by_name[op["table"]]
            assert op["left_on"] in known_cols
            assert op["right_on"] in {c.name for c in right.columns}
            known_cols.update(c.name for c in right.columns if c.name != op["right_on"])
        elif op["op"] == "filter":
            assert op["column"] in known_cols
        elif op["op"] == "select":
            assert set(op["columns"]).issubset(known_cols)
            assert len(op["columns"]) == len(set(op["columns"]))
            known_cols = set(op["columns"])
        elif op["op"] == "sort":
            columns = sort_columns(op)
            assert set(columns).issubset(known_cols)
            assert len(columns) == len(set(columns))
        elif op["op"] == "mutate":
            assert op["expr"]["source"] in known_cols
            assert not is_reserved_output_name(op["column"])
            known_cols.add(op["column"])
        elif op["op"] == "groupby":
            assert set(op["keys"]).issubset(known_cols)
            assert len(op["keys"]) == len(set(op["keys"]))
            for agg in op["aggs"]:
                assert agg["column"] in known_cols
            aliases = [agg["as"] for agg in op["aggs"]]
            assert len(aliases) == len(set(aliases))
            assert all(not is_reserved_output_name(alias) for alias in aliases)
            known_cols = set(op["keys"]) | {agg["as"] for agg in op["aggs"]}
        elif op["op"] == "aggregate":
            for agg in op["aggs"]:
                assert agg["column"] in known_cols
            aliases = [agg["as"] for agg in op["aggs"]]
            assert len(aliases) == len(set(aliases))
            assert all(not is_reserved_output_name(alias) for alias in aliases)
            known_cols = {agg["as"] for agg in op["aggs"]}


def test_generate_case_is_deterministic():
    a = generate_case(123).to_dict()
    b = generate_case(123).to_dict()
    assert a == b
    assert a["case_id"] == "case-00000123"
    assert a["tables"][0]["rows"] or a["tables"][0]["columns"]


def test_generate_case_edge_float_profile_is_supported():
    case = generate_case(123, profile="edge_float")
    assert case.case_id == "case-00000123"
    assert case.program.operations


def test_generate_case_workflow_profile_is_supported_and_valid():
    case = generate_case(123, profile="workflow")
    assert case.case_id.startswith("case-00000123-workflow-")
    assert case.program.operations
    assert validate_case_program(case) == []


def test_workflow_profile_covers_named_workflow_families():
    families = set()
    for seed in range(10):
        case = generate_case(seed, profile="workflow")
        families.add(case.case_id.rsplit("-", 1)[-1])
        assert validate_case_program(case) == []
    assert families == {"etl", "log", "feature", "join", "null"}


def test_generate_case_bughunt_profile_is_supported_and_valid():
    case = generate_case(123, profile="bughunt")
    assert case.case_id == "case-00000123-bughunt"
    assert case.program.operations
    assert validate_case_program(case) == []


def test_bughunt_profile_covers_per_column_sort_null_order():
    cases = [generate_case(seed, profile="bughunt") for seed in range(50)]
    sort_ops = [
        op
        for case in cases
        for op in case.program.operations
        if op.get("op") == "sort" and "keys" in op
    ]

    assert sort_ops
    assert any(any(key["nulls"] == "first" for key in op["keys"]) for op in sort_ops)
    assert all(validate_case_program(case) == [] for case in cases)


def test_bughunt_profile_mixes_issue_inspired_templates():
    cases = [generate_case(seed, profile="bughunt") for seed in range(120)]
    features = [extract_case_features(case) for case in cases]
    mixed_profiles = {case.metadata.get("mixed_generator_profile") for case in cases}

    assert any("pattern:join_null_key_topk" in item for item in features)
    assert any("pattern:empty_filter_groupby" in item for item in features)
    assert {
        "join_null_truth_filter",
        "global_null_aggregate",
        "string_count_groupby",
        "unique_count_groupby",
        "set_membership_filter",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "wide_offset_topk",
        "join_null_key_topk",
        "empty_filter_groupby",
        "ordered_groupby_sort",
        "topk_resort",
        "join_ordered_agg_topk",
    }.issubset(mixed_profiles)
    assert all(validate_case_program(case) == [] for case in cases)
    assert all(case.metadata.get("generator_profile", "bughunt") == "bughunt" for case in cases)


def test_generate_case_bughunt_no_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="bughunt_no_groupby")
    assert case.case_id == "case-00000123-bughunt-no-groupby"
    assert case.program.operations
    assert all(op["op"] != "groupby" for op in case.program.operations)
    assert validate_case_program(case) == []


def test_bughunt_no_groupby_profile_excludes_groupby_without_type_aware_generation():
    cases = [generate_case(seed, type_aware=False, profile="bughunt_no_groupby") for seed in range(100)]

    assert all("groupby" not in {op["op"] for op in case.program.operations} for case in cases)


def test_generate_case_null_groupby_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="null_groupby_topk")
    assert case.case_id == "case-00000123-null-groupby-topk"
    assert [op["op"] for op in case.program.operations] == ["mutate", "groupby", "select", "sort", "limit"]
    assert any(row["s"] is None for row in case.tables[0].rows)
    assert validate_case_program(case) == []


def test_generate_case_null_agg_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="null_agg_topk")
    assert case.case_id == "case-00000123-null-agg-topk"
    assert [op["op"] for op in case.program.operations] == ["groupby", "select", "sort", "limit"]
    assert all(row["g"] is not None for row in case.tables[0].rows)
    assert any(row["x"] is None for row in case.tables[0].rows)
    agg = case.program.operations[0]["aggs"][0]
    sort = case.program.operations[2]
    assert agg["func"] in {"min", "max"}
    assert case.program.operations[1]["columns"] == [agg["as"]]
    assert sort["columns"] == [agg["as"]]
    assert sort["ascending"] is (agg["func"] == "min")
    assert validate_case_program(case) == []


def test_generate_case_filter_null_agg_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="filter_null_agg_topk")
    assert case.case_id == "case-00000123-filter-null-agg-topk"
    assert [op["op"] for op in case.program.operations] == [
        "filter",
        "mutate",
        "select",
        "groupby",
        "select",
        "sort",
        "limit",
    ]
    assert case.program.operations[0] == {"op": "filter", "column": "lane", "cmp": "!=", "value": "skip"}
    assert case.program.operations[1] == {
        "op": "mutate",
        "column": "m_0",
        "expr": {"kind": "add_const", "source": "x", "value": 0},
    }
    assert case.program.operations[2]["columns"] == ["g", "m_0"]
    agg = case.program.operations[3]["aggs"][0]
    sort = case.program.operations[5]
    assert agg["func"] in {"min", "max"}
    assert case.program.operations[4]["columns"] == [agg["as"]]
    assert sort["columns"] == [agg["as"]]
    assert sort["ascending"] is (agg["func"] == "min")
    assert any(row["x"] is None for row in case.tables[0].rows if row["g"] == "a")
    assert validate_case_program(case) == []


def test_generate_case_join_null_agg_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_null_agg_topk")
    assert case.case_id == "case-00000123-join-null-agg-topk"
    assert [op["op"] for op in case.program.operations] == ["join", "mutate", "groupby", "select", "sort", "limit"]
    assert case.program.operations[0]["how"] == "left"
    agg = case.program.operations[2]["aggs"][0]
    sort = case.program.operations[4]
    assert agg["func"] in {"min", "max"}
    assert case.program.operations[3]["columns"] == [agg["as"]]
    assert sort["columns"] == [agg["as"]]
    assert sort["ascending"] is (agg["func"] == "min")
    assert validate_case_program(case) == []


def test_generate_case_join_null_key_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_null_key_topk")

    assert case.case_id == "case-00000123-join-null-key-topk"
    assert [op["op"] for op in case.program.operations] == ["join", "groupby", "select", "sort", "limit"]
    assert case.program.operations[0]["how"] == "left"
    assert case.program.operations[1]["keys"] == ["j"]
    assert case.program.operations[2]["columns"] == ["j"]
    assert "pattern:join_null_key_topk" in extract_case_features(case)
    assert validate_case_program(case) == []


def test_generate_case_wide_offset_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="wide_offset_topk")

    assert case.case_id == "case-00000123-wide-offset-topk"
    assert [op["op"] for op in case.program.operations] == ["sort", "offset", "limit"]
    assert len(case.tables[0].columns) >= 20
    assert "pattern:wide_offset_topk" in extract_case_features(case)
    assert validate_case_program(case) == []


def test_generate_case_empty_filter_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="empty_filter_groupby")

    assert case.case_id == "case-00000123-empty-filter-groupby"
    assert [op["op"] for op in case.program.operations] == ["filter", "groupby", "select", "sort", "limit"]
    assert case.program.operations[0] == {"op": "filter", "column": "lane", "cmp": "==", "value": "missing"}
    features = extract_case_features(case)
    assert "filter:empty-output" in features
    assert "pattern:empty_filter_groupby" in features
    assert validate_case_program(case) == []


def test_generate_case_join_filter_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_filter_groupby")
    assert case.case_id == "case-00000123-join-filter-groupby"
    assert [op["op"] for op in case.program.operations] == [
        "join",
        "filter",
        "mutate",
        "mutate",
        "groupby",
        "select",
        "sort",
        "limit",
    ]
    assert case.program.operations[0]["how"] == "inner"
    assert case.program.operations[1]["column"] == "j"
    assert len(case.program.operations[4]["aggs"]) == 3
    assert validate_case_program(case) == []


def test_generate_case_join_null_truth_filter_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_null_truth_filter")

    assert case.case_id == "case-00000123-join-null-truth-filter"
    assert [op["op"] for op in case.program.operations] == ["join", "filter", "select", "sort"]
    assert case.program.operations[0]["how"] == "left"
    assert case.program.operations[1]["cmp"] == "gt_is_not_true"
    assert case.program.operations[1]["column"] == "j"
    assert "pattern:join_null_truth_filter" in extract_case_features(case)
    left_ids = {row["id"] for row in case.tables[0].rows}
    right_ids = {row["id"] for row in case.tables[1].rows}
    assert left_ids - right_ids
    assert not any(row["id"] is None for table in case.tables for row in table.rows)
    assert validate_case_program(case) == []


def test_generate_case_join_groupby_stress_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_groupby_stress")

    assert case.case_id == "case-00000123-join-groupby-stress"
    assert [op["op"] for op in case.program.operations] == [
        "join",
        "groupby",
        "join",
        "groupby",
        "aggregate",
    ]
    assert [len(table.rows) for table in case.tables] == [99999, 99999, 99999]
    assert validate_case_program(case) == []


def test_generate_case_storage_offset_profile_is_supported_and_valid():
    even = generate_case(22656, profile="storage_offset")
    odd = generate_case(22657, profile="storage_offset")

    assert even.case_id == "case-00022656-storage-offset"
    assert [op["op"] for op in even.program.operations] == ["sort", "offset"]
    assert even.program.operations[0]["columns"] == ["id"]
    assert even.program.operations[1] == {"op": "offset", "n": 0}
    assert even.metadata["row_count"] == 300_000
    assert len(even.tables[0].rows) == 300_000
    assert odd.program.operations[1] == {"op": "offset", "n": 200_000}
    assert validate_case_program(even) == []
    assert validate_case_program(odd) == []


def test_null_agg_topk_profile_covers_min_asc_and_max_desc():
    pairs = set()
    for seed in range(50):
        case = generate_case(seed, profile="null_agg_topk")
        agg = case.program.operations[0]["aggs"][0]
        sort = case.program.operations[2]
        pairs.add((agg["func"], sort["ascending"]))
    assert ("min", True) in pairs
    assert ("max", False) in pairs


def test_generate_case_float_group_key_profile_is_supported_and_valid():
    case = generate_case(123, profile="float_group_key")
    assert case.case_id == "case-00000123-float-group-key"
    assert [op["op"] for op in case.program.operations] == [
        "join",
        "mutate",
        "filter",
        "sort",
        "mutate",
        "mutate",
        "mutate",
        "groupby",
    ]
    assert case.program.operations[-1]["keys"] == ["m_3"]
    assert validate_case_program(case) == []


def test_generate_case_join_null_sort_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_null_sort")
    assert case.case_id == "case-00000123-join-null-sort"
    assert [op["op"] for op in case.program.operations] == [
        "join",
        "mutate",
        "mutate",
        "select",
        "sort",
        "limit",
    ]
    assert case.program.operations[0]["how"] == "left"
    assert all(row["id"] is not None for row in case.tables[0].rows)
    assert all(row["id"] is not None for row in case.tables[1].rows)
    assert any(row["j"] is None for row in case.tables[1].rows)
    assert validate_case_program(case) == []


def test_generate_case_ordered_groupby_sort_profile_is_supported_and_valid():
    case = generate_case(123, profile="ordered_groupby_sort")

    assert case.case_id == "case-00000123-ordered-groupby-sort"
    assert [op["op"] for op in case.program.operations] == ["sort", "groupby", "select", "sort"]
    assert case.program.order_sensitive is True
    assert "pattern:ordered_groupby_sort" in extract_case_features(case)
    assert validate_case_program(case) == []


def test_generate_case_topk_resort_profile_is_supported_and_valid():
    case = generate_case(123, profile="topk_resort")

    assert case.case_id == "case-00000123-topk-resort"
    assert [op["op"] for op in case.program.operations] == ["sort", "limit", "sort", "offset"]
    assert case.program.order_sensitive is True
    assert "pattern:topk_resort" in extract_case_features(case)
    assert validate_case_program(case) == []


def test_topk_resort_profile_avoids_duplicate_sort_columns():
    for seed in range(120):
        case = generate_case(seed, profile="topk_resort")
        for op in case.program.operations:
            if op["op"] == "sort":
                columns = sort_columns(op)
                assert len(columns) == len(set(columns))
        assert validate_case_program(case) == []


def test_generate_case_join_ordered_agg_topk_profile_is_supported_and_valid():
    case = generate_case(123, profile="join_ordered_agg_topk")

    assert case.case_id == "case-00000123-join-ordered-agg-topk"
    assert [op["op"] for op in case.program.operations] == ["join", "sort", "groupby", "select", "sort", "limit"]
    assert case.program.order_sensitive is True
    assert "pattern:join_ordered_agg_topk" in extract_case_features(case)
    assert validate_case_program(case) == []


def test_generate_case_global_null_aggregate_profile_is_supported_and_valid():
    empty_case = generate_case(124, profile="global_null_aggregate")
    all_null_case = generate_case(125, profile="global_null_aggregate")

    for case in [empty_case, all_null_case]:
        assert case.case_id.endswith("-global-null-aggregate")
        assert [op["op"] for op in case.program.operations] == ["aggregate", "sort", "limit"]
        assert "pattern:global_null_aggregate" in extract_case_features(case)
        assert "op:aggregate" in extract_case_features(case)
        assert validate_case_program(case) == []

    assert empty_case.metadata["empty_input"] is True
    assert all_null_case.metadata["empty_input"] is False


def test_generate_case_string_count_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="string_count_groupby")
    features = extract_case_features(case)

    assert case.case_id == "case-00000123-string-count-groupby"
    assert [op["op"] for op in case.program.operations] == ["groupby", "sort", "limit"]
    assert "pattern:string_count_groupby" in features
    assert "agg:count:str" in features
    assert "groupby:null-key" in features
    assert validate_case_program(case) == []


def test_generate_case_unique_count_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="unique_count_groupby")
    features = extract_case_features(case)

    assert case.case_id == "case-00000123-unique-count-groupby"
    assert [op["op"] for op in case.program.operations] == ["groupby", "sort", "limit"]
    assert "pattern:unique_count_groupby" in features
    assert "agg:nunique:str" in features
    assert "agg:nunique:int" in features
    assert "groupby:null-key" in features
    assert case.metadata["source_issue"] == "https://github.com/apache/arrow/issues/36149"
    assert validate_case_program(case) == []


def test_generate_case_set_membership_filter_profile_is_supported_and_valid():
    case = generate_case(123, profile="set_membership_filter")
    features = extract_case_features(case)

    assert case.case_id == "case-00000123-set-membership-filter"
    assert [op["op"] for op in case.program.operations] == ["filter", "groupby", "sort", "limit"]
    assert case.program.operations[0] == {"op": "filter", "column": "s", "cmp": "in_set", "value": ["red", "", "中文"]}
    assert "pattern:set_membership_filter" in features
    assert "filter:set-membership" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/22149"
    assert validate_case_program(case) == []


def test_generate_case_null_predicate_filter_profile_is_supported_and_valid():
    even_case = generate_case(124, profile="null_predicate_filter")
    odd_case = generate_case(125, profile="null_predicate_filter")

    for case in [even_case, odd_case]:
        features = extract_case_features(case)
        assert case.case_id.endswith("-null-predicate-filter")
        assert [op["op"] for op in case.program.operations] == ["filter", "groupby", "sort", "limit"]
        assert case.program.operations[0]["cmp"] in {"is_null", "is_not_null"}
        assert "pattern:null_predicate_filter" in features
        assert "filter:null-predicate" in features
        assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/4978"
        assert validate_case_program(case) == []

    assert even_case.program.operations[0]["cmp"] == "is_null"
    assert odd_case.program.operations[0]["cmp"] == "is_not_null"


def test_generate_case_boolean_predicate_filter_profile_is_supported_and_valid():
    cases = [generate_case(seed, profile="boolean_predicate_filter") for seed in range(124, 128)]
    predicates = {case.program.operations[0]["cmp"] for case in cases}

    for case in cases:
        features = extract_case_features(case)
        assert case.case_id.endswith("-boolean-predicate-filter")
        assert [op["op"] for op in case.program.operations] == ["filter", "groupby", "sort", "limit"]
        assert case.program.operations[0]["column"] == "flag"
        assert case.program.operations[0]["value"] is None
        assert "pattern:boolean_predicate_filter" in features
        assert "filter:boolean-predicate" in features
        assert "filter:truth-test" in features
        assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/8516"
        assert validate_case_program(case) == []

    assert predicates == {"bool_is_true", "bool_is_not_true", "bool_is_false", "bool_is_not_false"}


def test_generate_case_post_topk_range_filter_profile_is_supported_and_valid():
    case = generate_case(124, profile="post_topk_range_filter")
    features = extract_case_features(case)

    assert case.case_id == "case-00000124-post-topk-range-filter"
    assert [op["op"] for op in case.program.operations] == ["sort", "limit", "filter", "select", "sort"]
    assert case.program.operations[2] == {"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 2]}
    assert "pattern:post_topk_range_filter" in features
    assert "pattern:range_filter" in features
    assert "filter:range-closed" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/26803"
    assert validate_case_program(case) == []


def test_generate_case_tuple_absence_filter_profile_is_supported_and_valid():
    case = generate_case(125, profile="tuple_absence_filter")
    features = extract_case_features(case)

    assert case.case_id == "case-00000125-tuple-absence-filter"
    assert [op["op"] for op in case.program.operations] == ["tuple_absence_filter", "select", "sort"]
    assert case.program.operations[0] == {
        "op": "tuple_absence_filter",
        "columns": ["a", "b"],
        "table": "t1",
        "right_columns": ["a", "b"],
    }
    assert "pattern:tuple_absence_filter" in features
    assert "filter:tuple-absence" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22418"
    assert validate_case_program(case) == []


def test_repair_operations_keeps_safe_string_aggregations_only():
    case = generate_case(123, profile="string_count_groupby")

    repaired = repair_operations(
        case.tables[0],
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "s", "func": "count", "as": "count_s"},
                    {"column": "s", "func": "nunique", "as": "uniq_s_count"},
                    {"column": "s", "func": "sum", "as": "sum_s"},
                    {"column": "x", "func": "sum", "as": "sum_x"},
                ],
            }
        ],
    )

    assert [agg["as"] for agg in repaired[0]["aggs"]] == ["count_s", "uniq_s_count", "sum_x"]


def test_bughunt_profile_biases_toward_multi_table_and_deeper_programs():
    cases = [generate_case(seed, profile="bughunt") for seed in range(50)]
    multi_table_count = sum(1 for case in cases if len(case.tables) > 1)
    avg_ops = sum(len(case.program.operations) for case in cases) / len(cases)
    assert multi_table_count >= 30
    assert avg_ops >= 3.0


def test_bughunt_profile_biases_toward_join_mutate_groupby_paths():
    cases = [generate_case(seed, profile="bughunt") for seed in range(100)]
    combined = 0
    covered_joins = 0
    for case in cases:
        ops = {op["op"] for op in case.program.operations}
        combined += int({"join", "mutate", "groupby"}.issubset(ops))
        if len(case.tables) > 1 and "join" in ops:
            left_ids = {row["id"] for row in case.tables[0].rows}
            right_ids = {row["id"] for row in case.tables[1].rows}
            covered_joins += int(left_ids.issubset(right_ids))
    assert combined >= 50
    assert covered_joins >= 50


def test_bughunt_no_groupby_profile_biases_join_mutate_filter_without_groupby():
    cases = [generate_case(seed, profile="bughunt_no_groupby") for seed in range(100)]
    assert all("groupby" not in {op["op"] for op in case.program.operations} for case in cases)
    joined = 0
    mut_filter = 0
    for case in cases:
        ops = {op["op"] for op in case.program.operations}
        joined += int("join" in ops)
        mut_filter += int({"mutate", "filter"}.issubset(ops))
    assert joined >= 50
    assert mut_filter >= 50


def test_bughunt_profile_injects_order_projection_probes():
    cases = [generate_case(seed, profile="bughunt") for seed in range(100)]
    probe_count = 0
    for case in cases:
        ops = case.program.operations
        assert validate_case_program(case) == []
        for idx in range(len(ops) - 1):
            if ops[idx].get("op") != "sort" or ops[idx + 1].get("op") != "select":
                continue
            sort_key = sort_columns(ops[idx])[0]
            if sort_key not in set(ops[idx + 1].get("columns", [])):
                probe_count += 1
                break

    assert probe_count >= 10


def test_repair_preserves_limit_after_sort_select_projection():
    case = generate_case(7)
    table = case.tables[0]
    repaired = repair_operations(
        table,
        [
            {"op": "sort", "columns": ["id"], "ascending": False},
            {"op": "select", "columns": ["g"]},
            {"op": "limit", "n": 1},
        ],
    )

    assert [op["op"] for op in repaired] == ["sort", "select", "limit"]


def test_repair_drops_groupby_aggregation_alias_colliding_with_key():
    case = generate_case(7)
    table = case.tables[0]
    repaired = repair_operations(
        table,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "max", "as": "g"},
                    {"column": "x", "func": "min", "as": "min_x"},
                ],
            },
        ],
    )

    assert repaired == [
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [{"column": "x", "func": "min", "as": "min_x"}],
        }
    ]


def test_repair_drops_reserved_aggregation_aliases():
    case = generate_case(7)
    table = case.tables[0]

    repaired = repair_operations(
        table,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "max", "as": "select"},
                    {"column": "x", "func": "min", "as": "min_x"},
                ],
            },
            {
                "op": "aggregate",
                "aggs": [
                    {"column": "min_x", "func": "max", "as": "where"},
                    {"column": "min_x", "func": "min", "as": "min_min_x"},
                ],
            },
        ],
    )

    assert repaired == [
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [{"column": "x", "func": "min", "as": "min_x"}],
        },
        {
            "op": "aggregate",
            "aggs": [{"column": "min_x", "func": "min", "as": "min_min_x"}],
        },
    ]


def test_make_safe_output_name_avoids_keywords_and_duplicates():
    assert make_safe_output_name("select") == "derived_select"
    assert make_safe_output_name("sum_x", used={"sum_x"}) == "sum_x_1"


def test_generated_program_uses_existing_columns_initially():
    case = generate_case(7)
    _assert_program_columns_are_valid(case)


def test_generated_programs_are_valid_for_seed_range():
    for seed in range(100):
        case = generate_case(seed)
        _assert_program_columns_are_valid(case)


def test_generator_can_emit_join_and_new_expressions():
    seen_ops = set()
    seen_exprs = set()
    for seed in range(300):
        case = generate_case(seed)
        seen_ops.update(op["op"] for op in case.program.operations)
        for op in case.program.operations:
            if op["op"] == "mutate":
                seen_exprs.add(op["expr"]["kind"])
    assert "join" in seen_ops
    assert {"arith_const", "string_length", "string_lower", "cast"} & seen_exprs


def test_common_and_workflow_profiles_do_not_emit_modulo_by_default():
    for seed in range(500):
        assert not _case_uses_modulo(generate_case(seed))
    for seed in range(20):
        assert not _case_uses_modulo(generate_case(seed, profile="workflow"))


def test_edge_float_profile_can_emit_modulo_boundary_cases():
    assert any(_case_uses_modulo(generate_case(seed, profile="edge_float")) for seed in range(300))


def _case_uses_modulo(case):
    return any(
        op.get("op") == "mutate"
        and op.get("expr", {}).get("kind") == "arith_const"
        and op.get("expr", {}).get("op") == "mod"
        for op in case.program.operations
    )
