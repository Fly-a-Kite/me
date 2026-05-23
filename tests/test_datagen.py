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
        elif op["op"] == "running_sum":
            assert op["source"] in known_cols
            assert all(key["column"] in known_cols for key in op["order_by"])
            assert not is_reserved_output_name(op["column"])
            known_cols.add(op["column"])
        elif op["op"] == "sortedness_check":
            assert op["column"] in known_cols
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "random_case_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "group_quantile_probe":
            assert not is_reserved_output_name(op["as"])
            assert op["values"] == [1, 2, 3]
            assert op["quantiles"] == [0.0, 0.5, 1.0]
            known_cols = {op["as"]}
        elif op["op"] == "scalar_subquery_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "window_avg_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "struct_distinct_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "bit_compare_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "round_even_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "series_rtruediv_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "uint64_isin_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "tuple_anti_null_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "sparse_mask_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "float_wrap_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "index_bool_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "empty_literal_groupby_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
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
        "running_sum_precision",
        "sortedness_null_placement",
        "simple_case_random_subject",
        "group_quantile_key_probe",
        "scalar_subquery_double_parentheses",
        "window_avg_rows_frame",
        "struct_distinct_unnest",
        "bit_compare_unequal_length",
        "round_even_float_scale",
        "series_rtruediv_operand_order",
        "pandas_uint64_isin_precision",
        "duckdb_tuple_anti_null_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
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


def test_generate_case_running_sum_precision_profile_is_supported_and_valid():
    case = generate_case(126, profile="running_sum_precision")
    features = extract_case_features(case)

    assert case.case_id == "case-00000126-running-sum-precision"
    assert [op["op"] for op in case.program.operations] == ["running_sum", "sort", "limit", "select"]
    assert case.program.operations[0] == {
        "op": "running_sum",
        "source": "x",
        "column": "run_x",
        "order_by": [{"column": "row_id", "ascending": True, "nulls": "last"}],
        "input_dtype": "float32",
    }
    assert "pattern:running_sum_precision" in features
    assert "running:float32" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/27662"
    assert validate_case_program(case) == []


def test_generate_case_sortedness_null_placement_profile_is_supported_and_valid():
    case = generate_case(127, profile="sortedness_null_placement")
    features = extract_case_features(case)

    assert case.case_id == "case-00000127-sortedness-null-placement"
    assert [op["op"] for op in case.program.operations] == ["sort", "sortedness_check"]
    assert case.program.operations[0] == {
        "op": "sort",
        "keys": [{"column": "x", "ascending": True, "nulls": "last"}],
    }
    assert case.program.operations[1] == {
        "op": "sortedness_check",
        "column": "x",
        "as": "sorted_ok_x",
        "ascending": True,
        "nulls": "first",
    }
    assert case.program.order_sensitive is True
    assert "pattern:sortedness_null_placement" in features
    assert "sortedness:null-placement-mismatch" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/26993"
    assert validate_case_program(case) == []


def test_generate_case_simple_case_random_subject_profile_is_supported_and_valid():
    case = generate_case(128, profile="simple_case_random_subject")
    features = extract_case_features(case)

    assert case.case_id == "case-00000128-simple-case-random-subject"
    assert case.program.operations == [
        {
            "op": "random_case_probe",
            "as": "unexpected_else_seen",
            "rows": 100_000,
            "branches": 3,
        }
    ]
    assert "pattern:simple_case_random_subject" in features
    assert "case_expr:random-subject" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22576"
    assert validate_case_program(case) == []


def test_generate_case_group_quantile_key_probe_profile_is_supported_and_valid():
    case = generate_case(129, profile="group_quantile_key_probe")
    features = extract_case_features(case)

    assert case.case_id == "case-00000129-group-quantile-key-probe"
    assert case.program.operations == [
        {
            "op": "group_quantile_probe",
            "as": "quantile_key_mismatch",
            "values": [1, 2, 3],
            "quantiles": [0.0, 0.5, 1.0],
        }
    ]
    assert "pattern:group_quantile_key_probe" in features
    assert "quantile:dynamic-key" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/25888"
    assert validate_case_program(case) == []


def test_generate_case_scalar_subquery_double_parentheses_profile_is_supported_and_valid():
    case = generate_case(130, profile="scalar_subquery_double_parentheses")
    features = extract_case_features(case)

    assert case.case_id == "case-00000130-scalar-subquery-double-parentheses"
    assert case.program.operations == [{"op": "scalar_subquery_probe", "as": "scalar_subquery_mismatch"}]
    assert "pattern:scalar_subquery_double_parentheses" in features
    assert "subquery:correlated-scalar" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/19851"
    assert validate_case_program(case) == []


def test_generate_case_window_avg_rows_frame_profile_is_supported_and_valid():
    case = generate_case(131, profile="window_avg_rows_frame")
    features = extract_case_features(case)

    assert case.case_id == "case-00000131-window-avg-rows-frame"
    assert case.program.operations == [{"op": "window_avg_probe", "as": "window_avg_mismatch"}]
    assert "pattern:window_avg_rows_frame" in features
    assert "window:rows-frame" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/26065"
    assert validate_case_program(case) == []


def test_generate_case_struct_distinct_unnest_profile_is_supported_and_valid():
    case = generate_case(132, profile="struct_distinct_unnest")
    features = extract_case_features(case)

    assert case.case_id == "case-00000132-struct-distinct-unnest"
    assert case.program.operations == [{"op": "struct_distinct_probe", "as": "struct_distinct_mismatch"}]
    assert "pattern:struct_distinct_unnest" in features
    assert "struct:unnest" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/17278"
    assert validate_case_program(case) == []


def test_generate_case_bit_compare_unequal_length_profile_is_supported_and_valid():
    case = generate_case(133, profile="bit_compare_unequal_length")
    features = extract_case_features(case)

    assert case.case_id == "case-00000133-bit-compare-unequal-length"
    assert case.program.operations == [{"op": "bit_compare_probe", "as": "bit_compare_mismatch"}]
    assert "pattern:bit_compare_unequal_length" in features
    assert "bit:unequal-length" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22527"
    assert validate_case_program(case) == []


def test_generate_case_round_even_float_scale_profile_is_supported_and_valid():
    case = generate_case(134, profile="round_even_float_scale")
    features = extract_case_features(case)

    assert case.case_id == "case-00000134-round-even-float-scale"
    assert case.program.operations == [{"op": "round_even_probe", "as": "round_even_mismatch"}]
    assert "pattern:round_even_float_scale" in features
    assert "numeric:round-even" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/19491"
    assert validate_case_program(case) == []


def test_generate_case_series_rtruediv_operand_order_profile_is_supported_and_valid():
    case = generate_case(135, profile="series_rtruediv_operand_order")
    features = extract_case_features(case)

    assert case.case_id == "case-00000135-series-rtruediv-operand-order"
    assert case.program.operations == [{"op": "series_rtruediv_probe", "as": "series_rtruediv_mismatch"}]
    assert "pattern:series_rtruediv_operand_order" in features
    assert "series:reverse-division" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/17760"
    assert validate_case_program(case) == []


def test_generate_case_pandas_uint64_isin_precision_profile_is_supported_and_valid():
    case = generate_case(136, profile="pandas_uint64_isin_precision")
    features = extract_case_features(case)

    assert case.case_id == "case-00000136-pandas-uint64-isin-precision"
    assert case.program.operations == [{"op": "uint64_isin_probe", "as": "uint64_isin_mismatch"}]
    assert "pattern:pandas_uint64_isin_precision" in features
    assert "pandas:uint64-isin" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/59609"
    assert validate_case_program(case) == []


def test_generate_case_duckdb_tuple_anti_null_semantics_profile_is_supported_and_valid():
    case = generate_case(137, profile="duckdb_tuple_anti_null_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000137-duckdb-tuple-anti-null-semantics"
    assert case.program.operations == [{"op": "tuple_anti_null_probe", "as": "tuple_anti_null_mismatch"}]
    assert "pattern:duckdb_tuple_anti_null_semantics" in features
    assert "duckdb:tuple-anti-null" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22418"
    assert validate_case_program(case) == []


def test_generate_case_pandas_sparse_array_mask_semantics_profile_is_supported_and_valid():
    case = generate_case(138, profile="pandas_sparse_array_mask_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000138-pandas-sparse-array-mask-semantics"
    assert case.program.operations == [{"op": "sparse_mask_probe", "as": "sparse_mask_mismatch"}]
    assert "pattern:pandas_sparse_array_mask_semantics" in features
    assert "pandas:sparse-mask" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/45284"
    assert validate_case_program(case) == []


def test_generate_case_polars_float_wrap_numerical_semantics_profile_is_supported_and_valid():
    case = generate_case(139, profile="polars_float_wrap_numerical_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000139-polars-float-wrap-numerical-semantics"
    assert case.program.operations == [{"op": "float_wrap_probe", "as": "float_wrap_mismatch"}]
    assert "pattern:polars_float_wrap_numerical_semantics" in features
    assert "polars:wrap-numerical" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/18546"
    assert validate_case_program(case) == []


def test_generate_case_pandas_index_bool_result_type_profile_is_supported_and_valid():
    case = generate_case(140, profile="pandas_index_bool_result_type")
    features = extract_case_features(case)

    assert case.case_id == "case-00000140-pandas-index-bool-result-type"
    assert case.program.operations == [{"op": "index_bool_probe", "as": "index_bool_mismatch"}]
    assert "pattern:pandas_index_bool_result_type" in features
    assert "pandas:index-bool" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/62766"
    assert validate_case_program(case) == []


def test_generate_case_polars_empty_literal_groupby_semantics_profile_is_supported_and_valid():
    case = generate_case(141, profile="polars_empty_literal_groupby_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000141-polars-empty-literal-groupby-semantics"
    assert case.program.operations == [
        {"op": "empty_literal_groupby_probe", "as": "empty_literal_groupby_mismatch"}
    ]
    assert "pattern:polars_empty_literal_groupby_semantics" in features
    assert "polars:empty-literal-groupby" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/23870"
    assert validate_case_program(case) == []


def test_generate_case_pandas_arrow_string_eq_sum_semantics_profile_is_supported_and_valid():
    case = generate_case(142, profile="pandas_arrow_string_eq_sum_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000142-pandas-arrow-string-eq-sum-semantics"
    assert case.program.operations == [
        {"op": "arrow_string_eq_sum_probe", "as": "arrow_string_eq_sum_mismatch"}
    ]
    assert "pattern:pandas_arrow_string_eq_sum_semantics" in features
    assert "pandas:arrow-string-eq-sum" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/63458"
    assert validate_case_program(case) == []


def test_generate_case_pandas_arrow_timestamp_loc_slice_semantics_profile_is_supported_and_valid():
    case = generate_case(143, profile="pandas_arrow_timestamp_loc_slice_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000143-pandas-arrow-timestamp-loc-slice-semantics"
    assert case.program.operations == [
        {"op": "arrow_timestamp_loc_slice_probe", "as": "arrow_timestamp_loc_slice_mismatch"}
    ]
    assert "pattern:pandas_arrow_timestamp_loc_slice_semantics" in features
    assert "pandas:arrow-timestamp-loc-slice" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/63526"
    assert validate_case_program(case) == []


def test_generate_case_pandas_arrow_timestamp_index_attr_semantics_profile_is_supported_and_valid():
    case = generate_case(144, profile="pandas_arrow_timestamp_index_attr_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000144-pandas-arrow-timestamp-index-attr-semantics"
    assert case.program.operations == [
        {"op": "arrow_timestamp_index_attr_probe", "as": "arrow_timestamp_index_attr_mismatch"}
    ]
    assert "pattern:pandas_arrow_timestamp_index_attr_semantics" in features
    assert "pandas:arrow-timestamp-index-attr" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/63527"
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
