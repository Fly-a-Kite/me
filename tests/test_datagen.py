from datadiff.case_policy import case_discovery_origin, replay_bug_filter_reason
from datadiff.config import DEFAULT_REPLAY_BUG_SOURCE_ISSUES
from datadiff.datagen import COMMON_API_WORKFLOW_TEMPLATES, generate_case, repair_operations
from datadiff.classification_oracle import validate_case_program
from datadiff.dsl import ColumnSpec, Program, TableData, sort_columns
from datadiff.guidance import extract_case_features
from datadiff.identifiers import is_reserved_output_name, make_safe_output_name
from datadiff.join_keys import join_key_columns


def _assert_program_columns_are_valid(case):
    table_by_name = {table.name: table for table in case.tables}
    known_cols = {c.name for c in case.tables[0].columns}
    for op in case.program.operations:
        if op["op"] == "join":
            right = table_by_name[op["table"]]
            left_keys = join_key_columns(op["left_on"])
            right_keys = join_key_columns(op["right_on"])
            assert len(left_keys) == len(right_keys)
            assert set(left_keys).issubset(known_cols)
            assert set(right_keys).issubset({c.name for c in right.columns})
            known_cols.update(c.name for c in right.columns if c.name not in set(right_keys))
        elif op["op"] == "union_all":
            right = table_by_name[op["table"]]
            assert known_cols.issubset({c.name for c in right.columns})
        elif op["op"] in {"semi_join", "anti_join"}:
            right = table_by_name[op["table"]]
            left_keys = join_key_columns(op["left_on"])
            right_keys = join_key_columns(op["right_on"])
            assert len(left_keys) == len(right_keys)
            assert set(left_keys).issubset(known_cols)
            assert set(right_keys).issubset({c.name for c in right.columns})
        elif op["op"] == "drop_nulls":
            assert set(op["columns"]).issubset(known_cols)
            assert len(op["columns"]) == len(set(op["columns"]))
        elif op["op"] == "filter":
            assert op["column"] in known_cols
        elif op["op"] == "select":
            assert set(op["columns"]).issubset(known_cols)
            assert len(op["columns"]) == len(set(op["columns"]))
            known_cols = set(op["columns"])
        elif op["op"] == "distinct":
            assert set(op["columns"]).issubset(known_cols)
            assert len(op["columns"]) == len(set(op["columns"]))
            known_cols = set(op["columns"])
        elif op["op"] == "fill_null":
            assert op["column"] in known_cols
        elif op["op"] == "coalesce":
            assert set(op["columns"]).issubset(known_cols)
            assert len(op["columns"]) == len(set(op["columns"]))
            assert len(op["columns"]) >= 2
            assert not is_reserved_output_name(op["as"])
            known_cols.add(op["as"])
        elif op["op"] == "case_when":
            assert op["condition"]["column"] in known_cols
            assert not is_reserved_output_name(op["as"])
            known_cols.add(op["as"])
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
        elif op["op"] == "row_number_filter":
            assert all(column in known_cols for column in op.get("partition_by", []))
            assert all(key["column"] in known_cols for key in op["order_by"])
            assert op["cmp"] in {"==", "<", "<="}
            assert op["value"] > 0
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
        elif op["op"] == "setop_all_duplicate_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "json_predicate_order_probe":
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
        elif op["op"] == "eval_inplace_alias_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "bool_reduction_skipna_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "dataset_isin_all_match_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "run_end_null_compute_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "large_string_partition_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "hash_pivot_wider_probe":
            assert not is_reserved_output_name(op["as"])
            known_cols = {op["as"]}
        elif op["op"] == "list_flatten_parent_indices_probe":
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


def test_bughunt_fresh_skips_known_replay_source_mixins():
    replay_mixed = generate_case(20, profile="bughunt")
    fresh = generate_case(20, profile="bughunt_fresh")
    organic_groupby = generate_case(1, profile="bughunt_fresh")

    assert replay_mixed.metadata["mixed_generator_profile"] == "wide_offset_topk"
    assert case_discovery_origin(replay_mixed) == "issue_inspired"
    assert fresh.metadata.get("mixed_generator_profile") != "wide_offset_topk"
    assert case_discovery_origin(fresh) == "organic"
    assert "groupby" in [op["op"] for op in organic_groupby.program.operations]


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
        "pyarrow_groupby_filter_cast_membership",
        "null_predicate_filter",
        "boolean_predicate_filter",
        "post_topk_range_filter",
        "tuple_absence_filter",
        "row_value_absence_filter",
        "wide_offset_topk",
        "join_null_key_topk",
        "empty_filter_groupby",
        "partitioned_running_sum",
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
        "duckdb_float_literal_precision",
        "polars_timestamp_precision_filter",
        "series_rtruediv_operand_order",
        "polars_reverse_division_columns",
        "pandas_uint64_isin_precision",
        "duckdb_tuple_anti_null_semantics",
        "datafusion_setop_all_duplicate_count",
        "duckdb_json_predicate_order_semantics",
        "pandas_sparse_array_mask_semantics",
        "polars_float_wrap_numerical_semantics",
        "pandas_index_bool_result_type",
        "polars_empty_literal_groupby_semantics",
        "pandas_arrow_string_eq_sum_semantics",
        "pandas_arrow_timestamp_loc_slice_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pandas_eval_inplace_aliasing_semantics",
        "pyarrow_dataset_isin_all_match_semantics",
        "pyarrow_large_string_partition_schema_semantics",
        "pyarrow_hash_pivot_wider_order_semantics",
        "pyarrow_list_flatten_parent_indices_semantics",
        "polars_rolling_mean_by_null_count_semantics",
        "csv_long_numeric_roundtrip",
    }.issubset(mixed_profiles)
    assert all(validate_case_program(case) == [] for case in cases)
    assert all(case.metadata.get("generator_profile", "bughunt") == "bughunt" for case in cases)


def test_generate_case_bughunt_no_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="bughunt_no_groupby")
    assert case.case_id == "case-00000123-bughunt-no-groupby"
    assert case.program.operations
    assert all(op["op"] != "groupby" for op in case.program.operations)
    assert validate_case_program(case) == []


def test_generate_case_issue_focus_profile_is_policy_gated_and_valid():
    cases = [generate_case(seed, profile="issue_focus") for seed in range(45)]
    mixed_profiles = {case.metadata.get("mixed_generator_profile") for case in cases}

    assert {"empty_filter_groupby", "row_value_absence_filter", "polars_reverse_division_columns"}.issubset(
        mixed_profiles
    )
    filter_reasons = set()
    for case in cases:
        assert case.metadata["generator_profile"] == "issue_focus"
        assert case.case_id.startswith(f"case-{case.seed:08d}-issue-focus-")
        assert validate_case_program(case) == []
        filter_reasons.add(
            replay_bug_filter_reason(
                case,
                enable_replay_bug=False,
                replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
            )
        )
    assert "" in filter_reasons
    assert {"known_replay_source_issue", "issue_replay_probe"}.issubset(filter_reasons)


def test_generate_case_deep_probe_rotation_profile_covers_deep_targets_with_policy_gate():
    cases = [generate_case(seed, profile="deep_probe_rotation") for seed in range(40)]
    mixed_profiles = {case.metadata.get("mixed_generator_profile") for case in cases}

    assert {
        "duckdb_json_predicate_order_semantics",
        "pandas_arrow_timestamp_index_attr_semantics",
        "pyarrow_run_end_null_compute_semantics",
        "polars_rolling_mean_by_null_count_semantics",
        "csv_long_numeric_roundtrip",
    }.issubset(mixed_profiles)
    origins = {case_discovery_origin(case) for case in cases}
    filter_reasons = {
        replay_bug_filter_reason(
            case,
            enable_replay_bug=False,
            replay_bug_source_issues=DEFAULT_REPLAY_BUG_SOURCE_ISSUES,
        )
        for case in cases
    }
    for case in cases:
        assert case.metadata["generator_profile"] == "deep_probe_rotation"
        assert case.case_id.startswith(f"case-{case.seed:08d}-deep-probe-rotation-")
        assert validate_case_program(case) == []
    assert {"organic", "issue_replay"}.issubset(origins)
    assert {"", "issue_replay_probe"}.issubset(filter_reasons)


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
    assert even.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22656"
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


def test_generate_case_bool_null_groupby_agg_profile_is_supported_and_valid():
    case = generate_case(123, profile="bool_null_groupby_agg")
    features = extract_case_features(case)

    assert case.case_id == "case-00000123-bool-null-groupby-agg"
    assert [op["op"] for op in case.program.operations] == ["groupby", "sort", "limit"]
    assert "pattern:bool_null_groupby_agg" in features
    assert "agg:boolean" in features
    assert "agg:any:bool" in features
    assert "agg:all:bool" in features
    assert "agg:min:bool" in features
    assert "agg:max:bool" in features
    assert "groupby:null-key" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/26671"
    assert validate_case_program(case) == []


def test_generate_case_large_int_filter_groupby_profile_is_supported_and_valid():
    case = generate_case(123, profile="large_int_filter_groupby")
    features = extract_case_features(case)

    assert case.case_id == "case-00000123-large-int-filter-groupby"
    assert [op["op"] for op in case.program.operations] == ["filter", "groupby", "sort", "limit"]
    assert "pattern:large_int_filter_groupby" in features
    assert "int:large-magnitude" in features
    assert "filter:set-membership" in features or "filter:range-closed" in features or "cmp:!=" in features
    assert "groupby:null-key" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/27726"
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


def test_generate_case_pyarrow_groupby_filter_cast_membership_profile_is_supported_and_valid():
    case = generate_case(123, profile="pyarrow_groupby_filter_cast_membership")
    features = extract_case_features(case)

    assert case.case_id == "case-00000123-pyarrow-groupby-filter-cast-membership"
    assert [op["op"] for op in case.program.operations] == ["groupby", "filter", "sort", "select"]
    filter_op = case.program.operations[1]
    assert filter_op["column"] == "agg_min_bucket"
    assert filter_op["cmp"] == "in_set"
    assert any(isinstance(value, float) and not value.is_integer() for value in filter_op["value"])
    assert "source_issue" not in case.metadata
    assert "pattern:pyarrow_groupby_filter_cast_membership" in features
    assert "membership:int-column-fractional-literal" in features
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


def test_generate_case_row_value_absence_filter_profile_is_supported_and_valid():
    case = generate_case(125, profile="row_value_absence_filter")
    features = extract_case_features(case)

    assert case.case_id == "case-00000125-row-value-absence-filter"
    assert [op["op"] for op in case.program.operations] == ["tuple_absence_filter", "select", "sort"]
    assert case.program.operations[0] == {
        "op": "tuple_absence_filter",
        "columns": ["left_a", "left_b"],
        "table": "t1",
        "right_columns": ["right_a", "right_b"],
    }
    assert "source_issue" not in case.metadata
    assert "pattern:row_value_absence_filter" in features
    assert "pattern:tuple_absence_filter" in features
    assert "filter:tuple-absence" in features
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


def test_generate_case_partitioned_running_sum_profile_is_supported_and_valid():
    case = generate_case(126, profile="partitioned_running_sum")
    features = extract_case_features(case)

    assert case.case_id == "case-00000126-partitioned-running-sum"
    assert [op["op"] for op in case.program.operations] == ["running_sum", "sort", "select"]
    assert case.program.operations[0]["partition_by"] == ["grp"]
    assert "pattern:partitioned_running_sum" in features
    assert "running:partitioned" in features
    assert validate_case_program(case) == []


def test_generate_case_path_basename_keyed_pick_profile_is_supported_and_valid():
    case = generate_case(107, profile="path_basename_keyed_pick")
    features = extract_case_features(case)

    assert case.case_id == "case-00000107-path-basename-keyed-pick"
    assert [op["op"] for op in case.program.operations] == [
        "mutate",
        "row_number_filter",
        "sort",
        "select",
    ]
    assert case.program.operations[0]["expr"] == {"kind": "string_basename", "source": "path_value"}
    assert case.program.operations[1]["partition_by"] == ["grp"]
    assert case.program.operations[1]["order_by"] == [
        {"column": "pick_key", "ascending": True, "nulls": "last"}
    ]
    assert case.program.order_sensitive is True
    assert case_discovery_origin(case) == "issue_replay"
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22849"
    assert "pattern:path_basename_keyed_pick" in features
    assert "expr:string_basename" in features
    assert "op:row_number_filter" in features
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


def test_generate_case_duckdb_float_literal_precision_profile_is_supported_and_valid():
    case = generate_case(135, profile="duckdb_float_literal_precision")
    features = extract_case_features(case)

    assert case.case_id == "case-00000135-duckdb-float-literal-precision"
    assert case.program.operations == [
        {
            "op": "float_literal_precision_probe",
            "as": "float_literal_precision_mismatch",
            "literal": "0.41000000000000003",
        }
    ]
    assert "pattern:duckdb_float_literal_precision" in features
    assert "float:literal-cast-consistency" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/22837"
    assert validate_case_program(case) == []


def test_generate_case_polars_timestamp_precision_filter_profile_is_supported_and_valid():
    case = generate_case(136, profile="polars_timestamp_precision_filter")
    features = extract_case_features(case)

    assert case.case_id == "case-00000136-polars-timestamp-precision-filter"
    assert case.program.operations == [
        {"op": "timestamp_precision_filter_probe", "as": "timestamp_precision_filter_mismatch"}
    ]
    assert "pattern:polars_timestamp_precision_filter" in features
    assert "timestamp:precision-filter" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/27726"
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


def test_generate_case_polars_reverse_division_columns_profile_is_supported_and_valid():
    case = generate_case(135, profile="polars_reverse_division_columns")
    features = extract_case_features(case)

    assert case.case_id == "case-00000135-polars-reverse-division-columns"
    assert [op["op"] for op in case.program.operations] == ["mutate", "select"]
    assert case.program.operations[0] == {
        "op": "mutate",
        "column": "ratio_value",
        "expr": {
            "kind": "reverse_division_columns",
            "source": "divisor_value",
            "numerator": "numerator_value",
        },
    }
    assert "source_issue" not in case.metadata
    assert "pattern:polars_reverse_division_columns" in features
    assert "arithmetic:reverse-division" in features
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


def test_generate_case_datafusion_setop_all_duplicate_count_profile_is_supported_and_valid():
    case = generate_case(137, profile="datafusion_setop_all_duplicate_count")
    features = extract_case_features(case)

    assert case.case_id == "case-00000137-datafusion-setop-all-duplicate-count"
    assert case.program.operations == [
        {"op": "setop_all_duplicate_probe", "as": "setop_all_duplicate_mismatch"}
    ]
    assert "pattern:datafusion_setop_all_duplicate_count" in features
    assert "datafusion:setop-all-duplicate-count" in features
    assert case.metadata["source_issue"] == "https://github.com/apache/datafusion/issues/12956"
    assert case.metadata["source_issue_alt"] == "https://github.com/apache/datafusion/issues/12955"
    assert validate_case_program(case) == []


def test_generate_case_duckdb_json_predicate_order_semantics_profile_is_supported_and_valid():
    case = generate_case(148, profile="duckdb_json_predicate_order_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000148-duckdb-json-predicate-order-semantics"
    assert case.program.operations == [
        {"op": "json_predicate_order_probe", "as": "json_predicate_order_mismatch"}
    ]
    assert "pattern:duckdb_json_predicate_order_semantics" in features
    assert "duckdb:json-predicate-order" in features
    assert case.metadata["source_issue"] == "https://github.com/duckdb/duckdb/issues/20366"
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


def test_generate_case_pandas_eval_inplace_aliasing_semantics_profile_is_supported_and_valid():
    case = generate_case(150, profile="pandas_eval_inplace_aliasing_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000150-pandas-eval-inplace-aliasing-semantics"
    assert case.program.operations == [
        {"op": "eval_inplace_alias_probe", "as": "eval_inplace_alias_mismatch"}
    ]
    assert "pattern:pandas_eval_inplace_aliasing_semantics" in features
    assert "pandas:eval-inplace-alias" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/65664"
    assert validate_case_program(case) == []


def test_generate_case_pandas_bool_reduction_skipna_semantics_profile_is_supported_and_valid():
    case = generate_case(151, profile="pandas_bool_reduction_skipna_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000151-pandas-bool-reduction-skipna-semantics"
    assert case.program.operations == [
        {"op": "bool_reduction_skipna_probe", "as": "bool_reduction_skipna_mismatch"}
    ]
    assert "pattern:pandas_bool_reduction_skipna_semantics" in features
    assert "pandas:bool-reduction-skipna" in features
    assert case.metadata["source_issue"] == "https://github.com/pandas-dev/pandas/issues/65710"
    assert validate_case_program(case) == []


def test_generate_case_pyarrow_dataset_isin_all_match_semantics_profile_is_supported_and_valid():
    case = generate_case(145, profile="pyarrow_dataset_isin_all_match_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000145-pyarrow-dataset-isin-all-match-semantics"
    assert case.program.operations == [
        {"op": "dataset_isin_all_match_probe", "as": "dataset_isin_all_match_mismatch"}
    ]
    assert "pattern:pyarrow_dataset_isin_all_match_semantics" in features
    assert "pyarrow:dataset-isin-all-match" in features
    assert case.metadata["source_issue"] == "https://github.com/apache/arrow/issues/46183"
    assert validate_case_program(case) == []


def test_generate_case_pyarrow_run_end_null_compute_semantics_profile_is_supported_and_valid():
    case = generate_case(146, profile="pyarrow_run_end_null_compute_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000146-pyarrow-run-end-null-compute-semantics"
    assert case.program.operations == [
        {"op": "run_end_null_compute_probe", "as": "run_end_null_compute_mismatch"}
    ]
    assert "pattern:pyarrow_run_end_null_compute_semantics" in features
    assert "pyarrow:run-end-null-compute" in features
    assert case.metadata["source_issue"] == "https://github.com/apache/arrow/issues/49889"
    assert validate_case_program(case) == []


def test_generate_case_pyarrow_large_string_partition_schema_profile_is_supported_and_valid():
    case = generate_case(147, profile="pyarrow_large_string_partition_schema_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000147-pyarrow-large-string-partition-schema-semantics"
    assert case.program.operations == [
        {"op": "large_string_partition_probe", "as": "large_string_partition_mismatch"}
    ]
    assert "pattern:pyarrow_large_string_partition_schema_semantics" in features
    assert "pyarrow:large-string-partition" in features
    assert case.metadata["source_issue"] == "https://github.com/apache/arrow/issues/47177"
    assert validate_case_program(case) == []


def test_generate_case_pyarrow_hash_pivot_wider_order_profile_is_supported_and_valid():
    case = generate_case(149, profile="pyarrow_hash_pivot_wider_order_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000149-pyarrow-hash-pivot-wider-order-semantics"
    assert case.program.operations == [
        {"op": "hash_pivot_wider_probe", "as": "hash_pivot_wider_mismatch"}
    ]
    assert "pattern:pyarrow_hash_pivot_wider_order_semantics" in features
    assert "pyarrow:hash-pivot-wider" in features
    assert case.metadata["source_issue"] == "https://github.com/apache/arrow/issues/48679"
    assert validate_case_program(case) == []


def test_generate_case_pyarrow_list_flatten_parent_indices_profile_is_supported_and_valid():
    case = generate_case(150, profile="pyarrow_list_flatten_parent_indices_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000150-pyarrow-list-flatten-parent-indices-semantics"
    assert case.program.operations == [
        {"op": "list_flatten_parent_indices_probe", "as": "list_flatten_parent_indices_mismatch"}
    ]
    assert "pattern:pyarrow_list_flatten_parent_indices_semantics" in features
    assert "pyarrow:list-flatten-parent-indices" in features
    assert "source_issue" not in case.metadata
    assert validate_case_program(case) == []


def test_generate_case_polars_rolling_mean_by_null_count_semantics_profile_is_supported_and_valid():
    case = generate_case(146, profile="polars_rolling_mean_by_null_count_semantics")
    features = extract_case_features(case)

    assert case.case_id == "case-00000146-polars-rolling-mean-by-null-count-semantics"
    assert case.program.operations == [
        {"op": "rolling_mean_by_null_count_probe", "as": "rolling_mean_by_null_count_mismatch"}
    ]
    assert "pattern:polars_rolling_mean_by_null_count_semantics" in features
    assert "polars:rolling-mean-by-null-count" in features
    assert case.metadata["source_issue"] == "https://github.com/pola-rs/polars/issues/27661"
    assert validate_case_program(case) == []


def test_generate_case_csv_long_numeric_roundtrip_profile_is_supported_and_fresh_safe():
    case = generate_case(153, profile="csv_long_numeric_roundtrip")
    features = extract_case_features(case)

    assert case.case_id == "case-00000153-csv-long-numeric-roundtrip"
    assert case.program.operations[0]["op"] == "csv_long_numeric_roundtrip_probe"
    assert case.program.operations[0]["as"] == "csv_long_numeric_roundtrip_mismatch"
    assert case.program.operations[0]["values"]
    assert "pattern:csv_long_numeric_roundtrip" in features
    assert "csv:long-numeric-roundtrip" in features
    assert "source_issue" not in case.metadata
    assert validate_case_program(case) == []


def test_generate_case_common_api_workflow_profile_covers_daily_low_complexity_ops():
    cases = [generate_case(seed, profile="common_api_workflow") for seed in range(len(COMMON_API_WORKFLOW_TEMPLATES))]
    templates = {case.metadata.get("workflow_template") for case in cases}
    op_sets = [{op["op"] for op in case.program.operations} for case in cases]

    assert templates == {
        "filter_mutate_project_topk",
        "membership_groupby_aggregate",
        "input_partition_union_groupby",
        "filter_input_materialization_groupby",
        "negative_membership_topk",
        "negative_membership_groupby",
        "string_derive_groupby",
        "string_upper_groupby",
        "string_upper_topk",
        "string_contains_groupby",
        "string_contains_topk",
        "string_strip_groupby",
        "string_strip_topk",
        "string_replace_groupby",
        "string_replace_topk",
        "string_slice_groupby",
        "string_slice_topk",
        "string_concat_groupby",
        "string_concat_topk",
        "string_contains_flag_groupby",
        "string_contains_flag_topk",
        "string_starts_with_flag_groupby",
        "string_starts_with_flag_topk",
        "string_ends_with_flag_groupby",
        "string_ends_with_flag_topk",
        "bool_not_groupby",
        "bool_not_topk",
        "join_filter_groupby",
        "nullable_bool_groupby",
        "bool_reduction_groupby_topk",
        "bool_reduction_global_summary",
        "ordered_slice_projection",
        "range_cast_groupby",
        "type_cast_boundary_groupby",
        "type_cast_boundary_topk",
        "abs_groupby",
        "abs_topk",
        "clip_groupby",
        "clip_topk",
        "double_filter_topk",
        "distinct_sort_topk",
        "distinct_null_topk",
        "filter_distinct_groupby",
        "fill_null_groupby",
        "fill_null_distinct_topk",
        "fill_null_filter_groupby_topk",
        "coalesce_fill_null_groupby_topk",
        "coalesce_groupby",
        "coalesce_topk",
        "case_when_classify_topk",
        "case_when_groupby",
        "union_all_filter_groupby",
        "union_all_case_when_topk",
        "drop_nulls_groupby",
        "union_all_drop_nulls_topk",
        "semi_join_filter_topk",
        "anti_join_groupby",
        "top_n_per_group",
        "dedup_latest_per_id",
        "running_total_by_group",
        "left_join_fill_groupby",
        "clean_key_join_groupby",
        "filtered_global_aggregate",
        "groupby_having_topk",
        "pagination_filter_select",
        "nunique_groupby",
        "join_nunique_groupby",
        "global_nunique_summary",
        "multi_key_groupby_summary",
        "multi_key_groupby_topk",
        "empty_filter_groupby_common",
        "empty_filter_global_aggregate",
        "multi_key_join_groupby",
        "multi_key_join_topk",
        "date_part_groupby",
        "date_part_topk",
        "string_split_part_groupby",
        "string_split_part_topk",
        "string_null_if_empty_groupby",
        "string_null_if_empty_topk",
        "string_length_groupby",
        "string_length_topk",
        "string_lower_groupby",
        "string_lower_topk",
        "string_basename_groupby",
        "string_basename_topk",
        "normalized_string_join_groupby",
        "normalized_string_join_topk",
        "normalized_string_semi_join_topk",
        "normalized_string_anti_join_groupby",
        "normalized_string_case_when_groupby",
        "normalized_string_case_when_topk",
        "string_pattern_case_when_groupby",
        "string_pattern_case_when_topk",
        "sql_distinct_null_coalesce_topk",
        "sql_left_join_coalesce_membership",
        "sql_case_membership_distinct_topk",
        "sql_union_coalesce_distinct_topk",
        "sql_left_join_case_membership_groupby",
        "sql_left_join_null_predicate_aggregate",
        "sql_coalesce_case_distinct_groupby",
        "sql_numeric_text_cast_membership_groupby",
        "sql_bool_membership_case_aggregate",
        "sql_left_join_bool_case_groupby",
        "sql_left_join_bool_coalesce_case_groupby",
        "sql_bool_antijoin_case_aggregate",
        "sql_left_join_bool_coalesce_filter_groupby",
        "sql_numeric_text_cast_bool_antijoin_groupby",
        "sql_multi_key_semijoin_case_groupby",
        "sql_multi_key_antijoin_case_groupby",
    }
    assert any({"join", "filter", "groupby"}.issubset(ops) for ops in op_sets)
    assert any({"sort", "offset", "limit", "select"}.issubset(ops) for ops in op_sets)
    assert any({"filter", "mutate", "groupby"}.issubset(ops) for ops in op_sets)
    assert any(
        case.metadata.get("workflow_template") == "input_partition_union_groupby"
        and case.program.op_sequence() == ["mutate", "fill_null", "groupby"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "filter_input_materialization_groupby"
        and case.program.op_sequence() == ["filter", "fill_null", "groupby"]
        for case in cases
    )
    assert any(
        op["op"] == "filter" and op.get("cmp") in {"str_contains", "str_starts_with", "str_ends_with"}
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "filter" and op.get("cmp") == "not_in_set"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_strip"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_upper"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_lower"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_basename"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_join_groupby"
        and case.program.op_sequence() == ["mutate", "mutate", "join", "fill_null", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_join_topk"
        and case.program.op_sequence() == ["mutate", "mutate", "join", "fill_null", "sort", "select", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_semi_join_topk"
        and case.program.op_sequence() == ["mutate", "mutate", "filter", "semi_join", "sort", "select", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_anti_join_groupby"
        and case.program.op_sequence() == ["mutate", "mutate", "filter", "anti_join", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_case_when_groupby"
        and case.program.op_sequence() == ["mutate", "mutate", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_case_when_topk"
        and case.program.op_sequence() == ["mutate", "mutate", "case_when", "sort", "select", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_pattern_case_when_groupby"
        and case.program.op_sequence() == ["mutate", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_pattern_case_when_topk"
        and case.program.op_sequence() == ["mutate", "case_when", "sort", "select", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_distinct_null_coalesce_topk"
        and case.program.op_sequence() == ["mutate", "coalesce", "distinct", "sort", "offset", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_coalesce_membership"
        and {"join", "coalesce", "fill_null"}.issubset(set(case.program.op_sequence()))
        and any(op["op"] in {"semi_join", "anti_join"} for op in case.program.operations)
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_case_membership_distinct_topk"
        and case.program.op_sequence() == ["mutate", "case_when", "distinct", "sort", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_union_coalesce_distinct_topk"
        and case.program.op_sequence() == ["union_all", "mutate", "coalesce", "distinct", "sort", "offset", "limit"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_case_membership_groupby"
        and case.program.op_sequence() == ["join", "case_when", "fill_null", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_null_predicate_aggregate"
        and case.program.op_sequence() == ["join", "filter", "aggregate"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_coalesce_case_distinct_groupby"
        and case.program.op_sequence() == ["mutate", "coalesce", "case_when", "distinct", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_numeric_text_cast_membership_groupby"
        and case.program.op_sequence() == ["mutate", "filter", "semi_join", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_bool_membership_case_aggregate"
        and case.program.op_sequence() == ["filter", "semi_join", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_bool_case_groupby"
        and case.program.op_sequence() == ["join", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_bool_coalesce_case_groupby"
        and case.program.op_sequence() == ["join", "coalesce", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_bool_antijoin_case_aggregate"
        and case.program.op_sequence() == ["filter", "anti_join", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_bool_coalesce_filter_groupby"
        and case.program.op_sequence() == ["join", "coalesce", "filter", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_numeric_text_cast_bool_antijoin_groupby"
        and case.program.op_sequence() == ["mutate", "filter", "filter", "anti_join", "case_when", "groupby", "sort"]
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_multi_key_semijoin_case_groupby"
        and case.program.op_sequence() == ["mutate", "filter", "semi_join", "case_when", "groupby", "sort"]
        and any(op["op"] == "semi_join" and isinstance(op.get("left_on"), list) for op in case.program.operations)
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_multi_key_antijoin_case_groupby"
        and case.program.op_sequence() == ["mutate", "filter", "anti_join", "case_when", "groupby", "sort"]
        and any(op["op"] == "anti_join" and isinstance(op.get("left_on"), list) for op in case.program.operations)
        for case in cases
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_replace"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_slice"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_split_part"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_null_if_empty"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_length"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_concat"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_contains"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_starts_with"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "string_ends_with"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "bool_not"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        agg.get("func") == "any"
        for case in cases
        for op in case.program.operations
        for agg in op.get("aggs", [])
    )
    assert any(
        agg.get("func") == "all"
        for case in cases
        for op in case.program.operations
        for agg in op.get("aggs", [])
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "clip"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "abs"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "date_part"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "cast" and op.get("expr", {}).get("to") == "int"
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "mutate" and op.get("expr", {}).get("kind") == "cast" and op.get("expr", {}).get("to") == "str"
        for case in cases
        for op in case.program.operations
    )
    assert any("distinct" in ops for ops in op_sets)
    assert any("fill_null" in ops for ops in op_sets)
    assert any("coalesce" in ops for ops in op_sets)
    assert any({"fill_null", "filter", "groupby", "sort", "limit"}.issubset(ops) for ops in op_sets)
    assert any({"coalesce", "fill_null", "groupby", "sort", "limit"}.issubset(ops) for ops in op_sets)
    assert any("case_when" in ops for ops in op_sets)
    assert any("union_all" in ops for ops in op_sets)
    assert any("drop_nulls" in ops for ops in op_sets)
    assert any("semi_join" in ops for ops in op_sets)
    assert any("anti_join" in ops for ops in op_sets)
    assert any("row_number_filter" in ops for ops in op_sets)
    assert any("running_sum" in ops for ops in op_sets)
    assert any("aggregate" in ops for ops in op_sets)
    assert any(
        op["op"] == "join" and isinstance(op.get("left_on"), list) and isinstance(op.get("right_on"), list)
        for case in cases
        for op in case.program.operations
    )
    assert any(
        op["op"] == "groupby" and len(op.get("keys", [])) > 1
        for case in cases
        for op in case.program.operations
    )
    assert any(
        case.metadata.get("workflow_template") == "empty_filter_global_aggregate"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "date_part_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "date_part_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_split_part_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_split_part_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_null_if_empty_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_null_if_empty_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_length_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_length_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_lower_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_lower_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_basename_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_basename_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_join_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_join_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_semi_join_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_anti_join_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_case_when_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "normalized_string_case_when_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_pattern_case_when_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "string_pattern_case_when_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_distinct_null_coalesce_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_coalesce_membership"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_case_membership_distinct_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_union_coalesce_distinct_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_case_membership_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_null_predicate_aggregate"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_coalesce_case_distinct_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_numeric_text_cast_membership_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_bool_membership_case_aggregate"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_bool_case_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_bool_coalesce_case_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_bool_antijoin_case_aggregate"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_left_join_bool_coalesce_filter_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_numeric_text_cast_bool_antijoin_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_multi_key_semijoin_case_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "sql_multi_key_antijoin_case_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "type_cast_boundary_groupby"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "type_cast_boundary_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "bool_reduction_groupby_topk"
        for case in cases
    )
    assert any(
        case.metadata.get("workflow_template") == "bool_reduction_global_summary"
        for case in cases
    )
    assert any(
        agg.get("func") == "nunique"
        for case in cases
        for op in case.program.operations
        for agg in op.get("aggs", [])
    )
    for case in cases:
        features = extract_case_features(case)
        assert case.case_id.endswith("-common-api-workflow")
        assert "pattern:common_api_workflow" in features
        assert "generator_profile:common_api_workflow" in features
        assert "source_issue" not in case.metadata
        assert case.metadata.get("discovery_origin") == "organic"
        assert validate_case_program(case) == []
        _assert_program_columns_are_valid(case)


def test_repair_operations_keeps_valid_semi_and_anti_join_only():
    case = generate_case(7)
    table = case.tables[0]
    lookup = TableData(
        "t_lookup",
        [ColumnSpec("id", "int"), ColumnSpec("bad", "str")],
        [{"id": 1, "bad": "one"}],
    )

    repaired = repair_operations(
        table,
        [
            {"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"},
            {"op": "anti_join", "table": "missing", "left_on": "id", "right_on": "id"},
            {"op": "anti_join", "table": "t_lookup", "left_on": "id", "right_on": "bad"},
        ],
        extra_tables=[lookup],
    )

    assert repaired == [{"op": "semi_join", "table": "t_lookup", "left_on": "id", "right_on": "id"}]


def test_repair_operations_keeps_valid_multi_key_semi_and_anti_join():
    case = generate_case(7)
    table = case.tables[0]
    lookup = TableData(
        "t_lookup",
        [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("bad", "str")],
        [{"id": 1, "g": "a", "bad": "one"}],
    )

    repaired = repair_operations(
        table,
        [
            {"op": "semi_join", "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["id", "g"]},
            {"op": "anti_join", "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["id"]},
            {"op": "anti_join", "table": "t_lookup", "left_on": ["id", "x"], "right_on": ["id", "bad"]},
        ],
        extra_tables=[lookup],
    )

    assert repaired == [
        {"op": "semi_join", "table": "t_lookup", "left_on": ["id", "g"], "right_on": ["id", "g"]}
    ]


def test_repair_operations_keeps_valid_coalesce_only():
    case = generate_case(7)
    table = case.tables[0]

    repaired = repair_operations(
        table,
        [
            {"op": "coalesce", "columns": ["x", "id"], "as": "x_or_id", "fallback": 0},
            {"op": "coalesce", "columns": ["g", "x"], "as": "bad_mixed"},
            {"op": "coalesce", "columns": ["g"], "as": "too_narrow"},
            {"op": "coalesce", "columns": ["x", "id"], "as": "select"},
        ],
    )

    assert repaired == [{"op": "coalesce", "columns": ["x", "id"], "as": "x_or_id", "fallback": 0}]


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


def test_repair_operations_accepts_typed_groupby_ops_without_mapping_method_conflict():
    case = generate_case(7)
    table = case.tables[0]
    typed_program = Program(
        "typed-groupby-repair",
        7,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [
                    {"column": "x", "func": "max", "as": "max_x"},
                    {"column": "x", "func": "min", "as": "min_x"},
                ],
            }
        ],
    )

    repaired = repair_operations(table, typed_program.operations)

    assert repaired == [
        {
            "op": "groupby",
            "keys": ["g"],
            "aggs": [
                {"column": "x", "func": "max", "as": "max_x"},
                {"column": "x", "func": "min", "as": "min_x"},
            ],
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
