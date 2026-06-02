import random

import datadiff.mutator as mutator_module
from datadiff.classification_oracle import validate_case_program
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.mutator import (
    DISCOVERY_MUTATION_OPERATOR_NAMES,
    MUTATION_OPERATOR_NAMES,
    PROBE_MUTATION_OPERATOR_NAMES,
    ROOT_TARGETED_MUTATION_OPERATOR_NAMES,
    SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES,
    mutation_operator_profiles,
    _append_group_quantile_probe,
    _append_scalar_subquery_probe,
    _append_window_avg_probe,
    _append_struct_distinct_probe,
    _append_bit_compare_probe,
    _append_round_even_probe,
    _append_series_rtruediv_probe,
    _append_uint64_isin_probe,
    _append_tuple_anti_null_probe,
    _append_setop_all_duplicate_probe,
    _append_json_predicate_order_probe,
    _append_sparse_mask_probe,
    _append_float_wrap_probe,
    _append_groupby_fractional_membership_filter,
    _append_normalized_string_membership,
    _append_sql_distinct_null_topk,
    _append_left_join_coalesce_membership,
    _append_left_join_case_membership,
    _append_empty_filter_global_aggregate,
    _append_sql_union_coalesce_distinct_topk,
    _append_boolean_membership_case_aggregate,
    _append_left_join_boolean_case_aggregate,
    _append_left_join_boolean_coalesce_case_aggregate,
    _append_boolean_antijoin_case_aggregate,
    _append_left_join_boolean_coalesce_filter_aggregate,
    _append_numeric_text_boolean_antijoin_case_aggregate,
    _append_multi_key_membership_case_aggregate,
    _append_index_bool_probe,
    _append_empty_literal_groupby_probe,
    _append_arrow_string_eq_sum_probe,
    _append_arrow_timestamp_loc_slice_probe,
    _append_arrow_timestamp_index_attr_probe,
    _append_eval_inplace_alias_probe,
    _append_dataset_isin_all_match_probe,
    _append_large_string_partition_probe,
    _append_hash_pivot_wider_probe,
    _append_list_flatten_parent_indices_probe,
    _append_rolling_mean_by_null_count_probe,
    _append_boolean_predicate_filter_probe,
    _append_grouped_topk_probe,
    _append_order_projection_probe,
    _append_range_filter_probe,
    _append_random_case_probe,
    _append_row_value_absence_filter,
    _append_running_sum_probe,
    _append_sortedness_check_probe,
    _append_truth_filter_probe,
    _append_tuple_absence_filter_probe,
    _fallback_column_type,
    _available_columns,
    _random_operation,
    mutate_case,
    mutate_case_with_metadata,
)


def test_random_operation_can_mutate_string_only_available_columns():
    table = TableData(
        "t0",
        [ColumnSpec("s", "str")],
        [{"s": "Alpha"}, {"s": "中文"}],
    )

    kinds = set()
    for seed in range(500):
        op = _random_operation([table], [], random.Random(seed))
        if op is None or op["op"] != "mutate":
            continue
        kinds.add(op["expr"]["kind"])
        assert op["expr"]["kind"] in {
            "string_length",
            "string_lower",
            "string_upper",
            "string_strip",
            "string_null_if_empty",
            "string_replace",
            "string_slice",
            "string_split_part",
            "string_basename",
            "string_contains",
            "string_starts_with",
            "string_ends_with",
        }
        assert op["expr"]["source"] == "s"
    assert "string_null_if_empty" in kinds
    assert "string_strip" in kinds
    assert "string_slice" in kinds
    assert "string_split_part" in kinds
    assert "string_basename" in kinds


def test_random_operation_can_append_date_part_for_date_like_string_column():
    table = TableData(
        "t0",
        [ColumnSpec("dt", "str")],
        [{"dt": "2024-01-03"}, {"dt": "2025-12-31"}],
    )

    seen = []
    for seed in range(500):
        op = _random_operation([table], [], random.Random(seed))
        if op is not None and op["op"] == "mutate":
            seen.append(op["expr"])

    assert any(expr["kind"] == "date_part" and expr["source"] == "dt" for expr in seen)


def test_mutation_operator_profiles_expose_semantic_family_affinity_alias():
    profile = mutation_operator_profiles(allow_probe_operators=False)["append_left_join_case_membership"]

    assert "conditional_semantics" in profile.semantic_family_affinity
    assert profile.semantic_affinity == profile.semantic_family_affinity


def test_available_columns_are_deduplicated_after_overwrite_and_select():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("g", "str")],
        [{"x": 1, "g": "Alpha"}],
    )

    available = _available_columns(
        [table],
        [
            {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "x", "value": 2}},
            {"op": "select", "columns": ["g", "m_1", "m_1"]},
        ],
    )

    assert available == ["g", "m_1"]


def test_mutate_case_with_metadata_records_lineage_and_operator():
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program("prog-base", 10, [{"op": "limit", "n": 2}]),
        metadata={"seed_lineage": {"root_seed": 3, "depth": 2}},
    )

    result = mutate_case_with_metadata(base, 99)

    assert result.case.case_id == "case-base-mut-99"
    assert result.case.metadata == result.metadata
    assert result.metadata["seed_lineage"] == {
        "root_seed": 3,
        "parent_seed": 10,
        "parent_case_id": "case-base",
        "mutation_seed": 99,
        "depth": 3,
    }
    assert result.metadata["mutation"]["operator"] in MUTATION_OPERATOR_NAMES
    assert isinstance(result.metadata["mutation"]["detail"], str)
    assert isinstance(result.metadata["mutation"]["changed"], bool)
    assert isinstance(mutate_case(base, 100), Case)


def test_mutate_case_with_metadata_retries_unproductive_operator(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    def fail(tables, operations, rnd):
        del tables, operations, rnd
        return "fail:no-key-or-bool-column"

    def good(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = 2
        return "good:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("fail", fail),
            mutator_module.MutationOperator("good", good),
        ),
    )

    result = mutate_case_with_metadata(base, 99)

    assert result.metadata["mutation"]["operator"] == "good"
    assert result.metadata["mutation"]["detail"] == "good:x"
    assert result.metadata["mutation"]["changed"] is True
    assert result.case.tables[0].rows[0]["x"] == 2


def test_mutate_case_prioritizes_operator_scores(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    def low_value(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = -1
        return "low:x"

    def high_value(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = 99
        return "high:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("low", low_value),
            mutator_module.MutationOperator("high", high_value),
        ),
    )

    result = mutate_case_with_metadata(base, 99, operator_scores={"low": -1.0, "high": 3.0})

    assert result.metadata["mutation"]["operator"] == "high"
    assert result.case.tables[0].rows[0]["x"] == 99


def test_mutate_case_uses_untried_operator_score(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    def known(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = -1
        return "known:x"

    def untried(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = 42
        return "untried:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("known", known),
            mutator_module.MutationOperator("untried", untried),
        ),
    )

    result = mutate_case_with_metadata(base, 99, operator_scores={"known": -1.0, "__untried__": 2.0})

    assert result.metadata["mutation"]["operator"] == "untried"
    assert result.case.tables[0].rows[0]["x"] == 42


def test_mutate_case_with_metadata_does_not_mutate_original_inputs(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 2]}]),
        metadata={"source_issue_alt": {"tags": ["alpha", "beta"]}},
    )

    def mutate_nested_inputs(tables, operations, rnd):
        del rnd
        tables[0].rows[0]["x"] = 99
        operations[0]["value"][0] = -10
        return "nested:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (mutator_module.MutationOperator("nested", mutate_nested_inputs),),
    )

    result = mutate_case_with_metadata(base, 99)

    assert base.tables[0].rows[0]["x"] == 1
    assert list(base.program.operations[0]["value"]) == [0, 2]
    assert result.case.tables[0].rows[0]["x"] == 99
    assert list(result.case.program.operations[0]["value"]) == [-10, 2]

    result.metadata["source_issue_alt"]["tags"].append("gamma")
    assert base.metadata["source_issue_alt"]["tags"] == ["alpha", "beta"]


def test_fallback_column_type_treats_probe_outputs_as_boolean():
    assert _fallback_column_type("unexpected_else_seen") == "bool"
    assert _fallback_column_type("sorted_ok_x") == "bool"
    assert _fallback_column_type("setop_all_duplicate_mismatch") == "bool"
    assert _fallback_column_type("rolling_mean_by_null_count_mismatch_1") == "bool"


def test_mutate_case_preserves_issue_profile_metadata_for_guidance():
    base = Case(
        "case-template",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program("prog-template", 10, [{"op": "limit", "n": 2}]),
        metadata={
            "generator_profile": "bughunt",
            "mixed_generator_profile": "post_topk_range_filter",
            "source_issue": "https://github.com/example/project/issues/1",
        },
    )

    result = mutate_case_with_metadata(base, 99)

    assert result.metadata["candidate_source"] == "feedback_mutation"
    assert result.metadata["generator_profile"] == "bughunt"
    assert result.metadata["mixed_generator_profile"] == "post_topk_range_filter"
    assert result.metadata["source_issue"] == "https://github.com/example/project/issues/1"


def test_mutate_case_can_exclude_probe_append_operators():
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("s", "str")], [{"x": 1, "s": "a"}, {"x": None, "s": "b"}])],
        Program("prog-base", 10, [{"op": "filter", "column": "x", "cmp": ">=", "value": 0}]),
    )

    seen = set()
    for seed in range(200):
        result = mutate_case_with_metadata(base, seed, allow_probe_operators=False)
        operator_name = result.metadata["mutation"]["operator"]
        seen.add(operator_name)
        assert operator_name not in PROBE_MUTATION_OPERATOR_NAMES
        assert operator_name not in ROOT_TARGETED_MUTATION_OPERATOR_NAMES
        assert operator_name not in SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES
        assert operator_name in DISCOVERY_MUTATION_OPERATOR_NAMES

    assert seen


def test_mutation_operator_registry_covers_row_value_and_operation_mutations():
    assert {"value", "nullify_value", "duplicate_row", "drop_row", "shuffle_rows"}.issubset(MUTATION_OPERATOR_NAMES)
    assert {"append_op", "drop_op", "tweak_op"}.issubset(MUTATION_OPERATOR_NAMES)
    assert "append_order_projection" in MUTATION_OPERATOR_NAMES
    assert "append_truth_filter" in MUTATION_OPERATOR_NAMES
    assert "append_boolean_predicate_filter" in MUTATION_OPERATOR_NAMES
    assert "append_range_filter" in MUTATION_OPERATOR_NAMES
    assert "append_tuple_absence_filter" in MUTATION_OPERATOR_NAMES
    assert "append_row_value_absence_filter" in MUTATION_OPERATOR_NAMES
    assert "append_row_value_absence_filter" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_running_sum" in MUTATION_OPERATOR_NAMES
    assert "append_sortedness_check" in MUTATION_OPERATOR_NAMES
    assert "append_random_case_probe" in MUTATION_OPERATOR_NAMES
    assert "append_group_quantile_probe" in MUTATION_OPERATOR_NAMES
    assert "append_scalar_subquery_probe" in MUTATION_OPERATOR_NAMES
    assert "append_window_avg_probe" in MUTATION_OPERATOR_NAMES
    assert "append_struct_distinct_probe" in MUTATION_OPERATOR_NAMES
    assert "append_bit_compare_probe" in MUTATION_OPERATOR_NAMES
    assert "append_round_even_probe" in MUTATION_OPERATOR_NAMES


def test_specialized_discovery_mutation_operator_alias_matches_compatibility_name():
    assert SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES == ROOT_TARGETED_MUTATION_OPERATOR_NAMES
    assert "append_series_rtruediv_probe" in MUTATION_OPERATOR_NAMES
    assert "append_uint64_isin_probe" in MUTATION_OPERATOR_NAMES
    assert "append_tuple_anti_null_probe" in MUTATION_OPERATOR_NAMES
    assert "append_setop_all_duplicate_probe" in MUTATION_OPERATOR_NAMES
    assert "append_json_predicate_order_probe" in MUTATION_OPERATOR_NAMES
    assert "append_sparse_mask_probe" in MUTATION_OPERATOR_NAMES
    assert "append_float_wrap_probe" in MUTATION_OPERATOR_NAMES
    assert "append_index_bool_probe" in MUTATION_OPERATOR_NAMES
    assert "append_empty_literal_groupby_probe" in MUTATION_OPERATOR_NAMES
    assert "append_arrow_string_eq_sum_probe" in MUTATION_OPERATOR_NAMES
    assert "append_arrow_timestamp_loc_slice_probe" in MUTATION_OPERATOR_NAMES
    assert "append_arrow_timestamp_index_attr_probe" in MUTATION_OPERATOR_NAMES
    assert "append_eval_inplace_alias_probe" in MUTATION_OPERATOR_NAMES
    assert "append_dataset_isin_all_match_probe" in MUTATION_OPERATOR_NAMES
    assert "append_large_string_partition_probe" in MUTATION_OPERATOR_NAMES
    assert "append_hash_pivot_wider_probe" in MUTATION_OPERATOR_NAMES
    assert "append_list_flatten_parent_indices_probe" in MUTATION_OPERATOR_NAMES
    assert "append_rolling_mean_by_null_count_probe" in MUTATION_OPERATOR_NAMES
    assert "append_grouped_topk" in MUTATION_OPERATOR_NAMES
    assert "append_groupby_fractional_membership_filter" in MUTATION_OPERATOR_NAMES
    assert "append_groupby_fractional_membership_filter" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_normalized_string_membership" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_sql_distinct_null_topk" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_coalesce_membership" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_case_membership" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_empty_filter_global_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_sql_union_coalesce_distinct_topk" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_boolean_membership_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_boolean_antijoin_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_filter_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_numeric_text_boolean_antijoin_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_multi_key_membership_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES


def test_mutation_operator_profiles_expose_semantic_affinity_for_high_risk_discovery_ops():
    profiles = mutation_operator_profiles(allow_probe_operators=False)

    left_join_case = profiles["append_left_join_case_membership"]
    normalized_membership = profiles["append_normalized_string_membership"]

    assert "conditional_semantics" in left_join_case.semantic_affinity
    assert "join_membership" in left_join_case.semantic_affinity
    assert "left_join_case_when_membership" in left_join_case.semantic_signal_affinity
    assert "string_semantics" in normalized_membership.semantic_affinity
    assert "normalized_string_membership_key" in normalized_membership.semantic_signal_affinity


def test_append_order_projection_mutation_drops_sort_key_but_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": 1, "s": "a"}],
    )
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_order_projection_probe([table], operations, random.Random(1))

    assert detail.startswith("append_order_projection:")
    sort_idx = next(idx for idx, op in enumerate(operations) if op["op"] == "sort")
    assert operations[sort_idx + 1]["op"] == "select"
    sort_op = operations[sort_idx]
    select_op = operations[sort_idx + 1]
    assert sort_op["keys"][0]["column"] not in set(select_op["columns"])
    case = Case("case-mut-order", 1, [table], Program("prog-mut-order", 1, operations))
    assert validate_case_program(case) == []


def test_append_truth_filter_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": None, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["id", "x"]}]

    detail = _append_truth_filter_probe([table], operations, random.Random(1))

    assert detail.startswith("append_truth_filter:")
    assert operations[-1]["op"] == "filter"
    assert operations[-1]["cmp"].endswith(("is_not_true", "is_not_false"))
    case = Case("case-mut-truth-filter", 1, [table], Program("prog-mut-truth-filter", 1, operations))
    assert validate_case_program(case) == []


def test_append_boolean_predicate_filter_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("flag", "bool"), ColumnSpec("s", "str")],
        [{"id": 0, "flag": True, "s": "b"}, {"id": 1, "flag": None, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["id", "flag"]}]

    detail = _append_boolean_predicate_filter_probe([table], operations, random.Random(1))

    assert detail.startswith("append_boolean_predicate_filter:")
    assert operations[-1]["op"] == "filter"
    assert operations[-1]["cmp"].startswith("bool_is_")
    assert operations[-1]["value"] is None
    case = Case("case-mut-bool-filter", 1, [table], Program("prog-mut-bool-filter", 1, operations))
    assert validate_case_program(case) == []


def test_append_range_filter_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": None, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["id", "x"]}]

    detail = _append_range_filter_probe([table], operations, random.Random(1))

    assert detail.startswith("append_range_filter:")
    assert operations[-1]["op"] == "filter"
    assert operations[-1]["cmp"] == "range_closed"
    assert len(operations[-1]["value"]) == 2
    case = Case("case-mut-range-filter", 1, [table], Program("prog-mut-range-filter", 1, operations))
    assert validate_case_program(case) == []


def test_append_tuple_absence_filter_mutation_stays_valid():
    left = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": None, "s": "a"}],
    )
    right = TableData(
        "t1",
        [ColumnSpec("id", "int"), ColumnSpec("j", "int"), ColumnSpec("tag", "str")],
        [{"id": 0, "j": None, "tag": "b"}],
    )
    operations = [{"op": "select", "columns": ["id", "x", "s"]}]

    detail = _append_tuple_absence_filter_probe([left, right], operations, random.Random(1))

    assert detail.startswith("append_tuple_absence_filter:")
    assert operations[-1]["op"] == "tuple_absence_filter"
    assert len(operations[-1]["columns"]) == 2
    assert operations[-1]["table"] == "t1"
    case = Case("case-mut-tuple-filter", 1, [left, right], Program("prog-mut-tuple-filter", 1, operations))
    assert validate_case_program(case) == []


def test_append_row_value_absence_filter_mutation_stays_valid_and_discoverable():
    left = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": None, "s": "a"}],
    )
    right = TableData(
        "t1",
        [ColumnSpec("id", "int"), ColumnSpec("j", "int"), ColumnSpec("tag", "str")],
        [{"id": 0, "j": None, "tag": "b"}],
    )
    operations = [{"op": "select", "columns": ["id", "x", "s"]}]

    detail = _append_row_value_absence_filter([left, right], operations, random.Random(1))

    assert detail.startswith("append_row_value_absence_filter:")
    assert operations[-1]["op"] == "tuple_absence_filter"
    assert len(operations[-1]["columns"]) == 2
    assert operations[-1]["table"] == "t1"
    case = Case("case-mut-row-value-filter", 1, [left, right], Program("prog-mut-row-value-filter", 1, operations))
    assert validate_case_program(case) == []


def test_append_running_sum_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("row_id", "int"), ColumnSpec("x", "float"), ColumnSpec("s", "str")],
        [{"row_id": 0, "x": 0.0005, "s": "a"}, {"row_id": 1, "x": 0.0005, "s": "b"}],
    )
    operations = [{"op": "select", "columns": ["row_id", "x"]}]

    detail = _append_running_sum_probe([table], operations, random.Random(1))

    assert detail.startswith("append_running_sum:")
    assert operations[-1]["op"] == "running_sum"
    assert operations[-1]["column"].startswith("run_")
    assert [key["column"] for key in operations[-1]["order_by"]] == ["row_id", "x"]
    case = Case("case-mut-running-sum", 1, [table], Program("prog-mut-running-sum", 1, operations))
    assert validate_case_program(case) == []


def test_append_sortedness_check_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": None, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["id", "x"]}]

    detail = _append_sortedness_check_probe([table], operations, random.Random(1))

    assert detail.startswith("append_sortedness_check:")
    assert [op["op"] for op in operations[-2:]] == ["sort", "sortedness_check"]
    assert operations[-1]["as"].startswith("sorted_ok_")
    case = Case("case-mut-sortedness", 1, [table], Program("prog-mut-sortedness", 1, operations))
    assert validate_case_program(case) == []


def test_append_random_case_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_random_case_probe([table], operations, random.Random(1))

    assert detail.startswith("append_random_case_probe:")
    assert operations[-1]["op"] == "random_case_probe"
    assert operations[-1]["as"] == "unexpected_else_seen"
    case = Case("case-mut-random-case", 1, [table], Program("prog-mut-random-case", 1, operations))
    assert validate_case_program(case) == []


def test_append_group_quantile_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_group_quantile_probe([table], operations, random.Random(1))

    assert detail.startswith("append_group_quantile_probe:")
    assert operations[-1] == {
        "op": "group_quantile_probe",
        "as": "quantile_key_mismatch",
        "values": [1, 2, 3],
        "quantiles": [0.0, 0.5, 1.0],
    }
    case = Case("case-mut-group-quantile", 1, [table], Program("prog-mut-group-quantile", 1, operations))
    assert validate_case_program(case) == []


def test_append_scalar_subquery_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_scalar_subquery_probe([table], operations, random.Random(1))

    assert detail.startswith("append_scalar_subquery_probe:")
    assert operations[-1] == {"op": "scalar_subquery_probe", "as": "scalar_subquery_mismatch"}
    case = Case("case-mut-scalar-subquery", 1, [table], Program("prog-mut-scalar-subquery", 1, operations))
    assert validate_case_program(case) == []


def test_append_window_avg_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_window_avg_probe([table], operations, random.Random(1))

    assert detail.startswith("append_window_avg_probe:")
    assert operations[-1] == {"op": "window_avg_probe", "as": "window_avg_mismatch"}
    case = Case("case-mut-window-avg", 1, [table], Program("prog-mut-window-avg", 1, operations))
    assert validate_case_program(case) == []


def test_append_struct_distinct_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_struct_distinct_probe([table], operations, random.Random(1))

    assert detail.startswith("append_struct_distinct_probe:")
    assert operations[-1] == {"op": "struct_distinct_probe", "as": "struct_distinct_mismatch"}
    case = Case("case-mut-struct-distinct", 1, [table], Program("prog-mut-struct-distinct", 1, operations))
    assert validate_case_program(case) == []


def test_append_bit_compare_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_bit_compare_probe([table], operations, random.Random(1))

    assert detail.startswith("append_bit_compare_probe:")
    assert operations[-1] == {"op": "bit_compare_probe", "as": "bit_compare_mismatch"}
    case = Case("case-mut-bit-compare", 1, [table], Program("prog-mut-bit-compare", 1, operations))
    assert validate_case_program(case) == []


def test_append_round_even_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_round_even_probe([table], operations, random.Random(1))

    assert detail.startswith("append_round_even_probe:")
    assert operations[-1] == {"op": "round_even_probe", "as": "round_even_mismatch"}
    case = Case("case-mut-round-even", 1, [table], Program("prog-mut-round-even", 1, operations))
    assert validate_case_program(case) == []


def test_append_series_rtruediv_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_series_rtruediv_probe([table], operations, random.Random(1))

    assert detail.startswith("append_series_rtruediv_probe:")
    assert operations[-1] == {"op": "series_rtruediv_probe", "as": "series_rtruediv_mismatch"}
    case = Case("case-mut-series-rtruediv", 1, [table], Program("prog-mut-series-rtruediv", 1, operations))
    assert validate_case_program(case) == []


def test_append_uint64_isin_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_uint64_isin_probe([table], operations, random.Random(1))

    assert detail.startswith("append_uint64_isin_probe:")
    assert operations[-1] == {"op": "uint64_isin_probe", "as": "uint64_isin_mismatch"}
    case = Case("case-mut-uint64-isin", 1, [table], Program("prog-mut-uint64-isin", 1, operations))
    assert validate_case_program(case) == []


def test_append_tuple_anti_null_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_tuple_anti_null_probe([table], operations, random.Random(1))

    assert detail.startswith("append_tuple_anti_null_probe:")
    assert operations[-1] == {"op": "tuple_anti_null_probe", "as": "tuple_anti_null_mismatch"}
    case = Case("case-mut-tuple-anti-null", 1, [table], Program("prog-mut-tuple-anti-null", 1, operations))
    assert validate_case_program(case) == []


def test_append_setop_all_duplicate_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_setop_all_duplicate_probe([table], operations, random.Random(1))

    assert detail.startswith("append_setop_all_duplicate_probe:")
    assert operations[-1] == {
        "op": "setop_all_duplicate_probe",
        "as": "setop_all_duplicate_mismatch",
    }
    case = Case(
        "case-mut-setop-all-duplicate",
        1,
        [table],
        Program("prog-mut-setop-all-duplicate", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_json_predicate_order_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("payload", "str")],
        [{"id": 0, "payload": "x"}],
    )
    operations = [{"op": "select", "columns": ["payload"]}]

    detail = _append_json_predicate_order_probe([table], operations, random.Random(1))

    assert detail.startswith("append_json_predicate_order_probe:")
    assert operations[-1] == {"op": "json_predicate_order_probe", "as": "json_predicate_order_mismatch"}
    case = Case("case-mut-json-predicate-order", 1, [table], Program("prog-mut-json-predicate-order", 1, operations))
    assert validate_case_program(case) == []


def test_append_sparse_mask_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_sparse_mask_probe([table], operations, random.Random(1))

    assert detail.startswith("append_sparse_mask_probe:")
    assert operations[-1] == {"op": "sparse_mask_probe", "as": "sparse_mask_mismatch"}
    case = Case("case-mut-sparse-mask", 1, [table], Program("prog-mut-sparse-mask", 1, operations))
    assert validate_case_program(case) == []


def test_append_float_wrap_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_float_wrap_probe([table], operations, random.Random(1))

    assert detail.startswith("append_float_wrap_probe:")
    assert operations[-1] == {"op": "float_wrap_probe", "as": "float_wrap_mismatch"}
    case = Case("case-mut-float-wrap", 1, [table], Program("prog-mut-float-wrap", 1, operations))
    assert validate_case_program(case) == []


def test_append_index_bool_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_index_bool_probe([table], operations, random.Random(1))

    assert detail.startswith("append_index_bool_probe:")
    assert operations[-1] == {"op": "index_bool_probe", "as": "index_bool_mismatch"}
    case = Case("case-mut-index-bool", 1, [table], Program("prog-mut-index-bool", 1, operations))
    assert validate_case_program(case) == []


def test_append_empty_literal_groupby_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int")],
        [{"id": 0, "x": 2}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_empty_literal_groupby_probe([table], operations, random.Random(1))

    assert detail.startswith("append_empty_literal_groupby_probe:")
    assert operations[-1] == {
        "op": "empty_literal_groupby_probe",
        "as": "empty_literal_groupby_mismatch",
    }
    case = Case("case-mut-empty-literal-groupby", 1, [table], Program("prog-mut-empty-literal-groupby", 1, operations))
    assert validate_case_program(case) == []


def test_append_arrow_string_eq_sum_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["s"]}]

    detail = _append_arrow_string_eq_sum_probe([table], operations, random.Random(1))

    assert detail.startswith("append_arrow_string_eq_sum_probe:")
    assert operations[-1] == {
        "op": "arrow_string_eq_sum_probe",
        "as": "arrow_string_eq_sum_mismatch",
    }
    case = Case("case-mut-arrow-string-eq-sum", 1, [table], Program("prog-mut-arrow-string-eq-sum", 1, operations))
    assert validate_case_program(case) == []


def test_append_arrow_timestamp_loc_slice_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["s"]}]

    detail = _append_arrow_timestamp_loc_slice_probe([table], operations, random.Random(1))

    assert detail.startswith("append_arrow_timestamp_loc_slice_probe:")
    assert operations[-1] == {
        "op": "arrow_timestamp_loc_slice_probe",
        "as": "arrow_timestamp_loc_slice_mismatch",
    }
    case = Case(
        "case-mut-arrow-timestamp-loc-slice",
        1,
        [table],
        Program("prog-mut-arrow-timestamp-loc-slice", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_arrow_timestamp_index_attr_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "s": "a"}],
    )
    operations = [{"op": "select", "columns": ["s"]}]

    detail = _append_arrow_timestamp_index_attr_probe([table], operations, random.Random(1))

    assert detail.startswith("append_arrow_timestamp_index_attr_probe:")
    assert operations[-1] == {
        "op": "arrow_timestamp_index_attr_probe",
        "as": "arrow_timestamp_index_attr_mismatch",
    }
    case = Case(
        "case-mut-arrow-timestamp-index-attr",
        1,
        [table],
        Program("prog-mut-arrow-timestamp-index-attr", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_dataset_isin_all_match_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "float")],
        [{"id": 0, "x": 0.0}],
    )
    operations = [{"op": "select", "columns": ["x"]}]

    detail = _append_dataset_isin_all_match_probe([table], operations, random.Random(1))

    assert detail.startswith("append_dataset_isin_all_match_probe:")
    assert operations[-1] == {
        "op": "dataset_isin_all_match_probe",
        "as": "dataset_isin_all_match_mismatch",
    }
    case = Case(
        "case-mut-dataset-isin-all-match",
        1,
        [table],
        Program("prog-mut-dataset-isin-all-match", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_large_string_partition_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("part_name", "str")],
        [{"id": 0, "part_name": "a"}],
    )
    operations = [{"op": "select", "columns": ["part_name"]}]

    detail = _append_large_string_partition_probe([table], operations, random.Random(1))

    assert detail.startswith("append_large_string_partition_probe:")
    assert operations[-1] == {
        "op": "large_string_partition_probe",
        "as": "large_string_partition_mismatch",
    }
    case = Case(
        "case-mut-large-string-partition",
        1,
        [table],
        Program("prog-mut-large-string-partition", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_eval_inplace_alias_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("nums", "int")],
        [{"id": 0, "nums": -1}],
    )
    operations = [{"op": "select", "columns": ["nums"]}]

    detail = _append_eval_inplace_alias_probe([table], operations, random.Random(1))

    assert detail.startswith("append_eval_inplace_alias_probe:")
    assert operations[-1] == {
        "op": "eval_inplace_alias_probe",
        "as": "eval_inplace_alias_mismatch",
    }
    case = Case(
        "case-mut-eval-inplace-alias",
        1,
        [table],
        Program("prog-mut-eval-inplace-alias", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_hash_pivot_wider_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("pivot_key", "str")],
        [{"id": 0, "pivot_key": "k"}],
    )
    operations = [{"op": "select", "columns": ["pivot_key"]}]

    detail = _append_hash_pivot_wider_probe([table], operations, random.Random(1))

    assert detail.startswith("append_hash_pivot_wider_probe:")
    assert operations[-1] == {
        "op": "hash_pivot_wider_probe",
        "as": "hash_pivot_wider_mismatch",
    }
    case = Case(
        "case-mut-hash-pivot-wider",
        1,
        [table],
        Program("prog-mut-hash-pivot-wider", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_list_flatten_parent_indices_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("probe_id", "int")],
        [{"id": 0, "probe_id": 0}],
    )
    operations = [{"op": "select", "columns": ["probe_id"]}]

    detail = _append_list_flatten_parent_indices_probe([table], operations, random.Random(1))

    assert detail.startswith("append_list_flatten_parent_indices_probe:")
    assert operations[-1] == {
        "op": "list_flatten_parent_indices_probe",
        "as": "list_flatten_parent_indices_mismatch",
    }
    case = Case(
        "case-mut-list-flatten-parent-indices",
        1,
        [table],
        Program("prog-mut-list-flatten-parent-indices", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_rolling_mean_by_null_count_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "float")],
        [{"id": 0, "x": 0.0}],
    )
    operations = [{"op": "select", "columns": ["x"]}]

    detail = _append_rolling_mean_by_null_count_probe([table], operations, random.Random(1))

    assert detail.startswith("append_rolling_mean_by_null_count_probe:")
    assert operations[-1] == {
        "op": "rolling_mean_by_null_count_probe",
        "as": "rolling_mean_by_null_count_mismatch",
    }
    case = Case(
        "case-mut-rolling-mean-by-null-count",
        1,
        [table],
        Program("prog-mut-rolling-mean-by-null-count", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_grouped_topk_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "int"), ColumnSpec("s", "str")],
        [{"id": 0, "x": 2, "s": "b"}, {"id": 1, "x": None, "s": "a"}],
    )
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_grouped_topk_probe([table], operations, random.Random(1))

    assert detail.startswith("append_grouped_topk:")
    assert [op["op"] for op in operations[-4:]] == ["groupby", "select", "sort", "limit"]
    case = Case("case-mut-grouped-topk", 1, [table], Program("prog-mut-grouped-topk", 1, operations))
    assert validate_case_program(case) == []


def test_append_groupby_fractional_membership_filter_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("bucket_code", "int"), ColumnSpec("amount_code", "int")],
        [
            {"bucket_code": 0, "amount_code": 2},
            {"bucket_code": 0, "amount_code": 10},
            {"bucket_code": 1, "amount_code": 2},
        ],
    )
    operations = [
        {
            "op": "groupby",
            "keys": ["bucket_code"],
            "aggs": [{"column": "bucket_code", "func": "min", "as": "agg_min_bucket"}],
        }
    ]

    detail = _append_groupby_fractional_membership_filter([table], operations, random.Random(1))

    assert detail.startswith("append_groupby_fractional_membership_filter:")
    assert operations[-1]["op"] == "filter"
    assert operations[-1]["column"] == "agg_min_bucket"
    assert operations[-1]["cmp"] == "in_set"
    assert any(isinstance(value, float) and not value.is_integer() for value in operations[-1]["value"])
    case = Case("case-mut-groupby-membership", 1, [table], Program("prog-mut-groupby-membership", 1, operations))
    assert validate_case_program(case) == []


def test_append_normalized_string_membership_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("s", "str"), ColumnSpec("x", "int")],
        [
            {"id": 0, "s": " Alpha ", "x": 1},
            {"id": 1, "s": "Beta", "x": 2},
            {"id": 2, "s": None, "x": 3},
            {"id": 3, "s": "space value", "x": 4},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_normalized_string_membership(tables, operations, random.Random(1))

    assert detail.startswith("append_normalized_string_membership:")
    assert any(op["op"] in {"semi_join", "anti_join"} for op in operations)
    assert any(t.name.startswith("t_string_membership_mut") for t in tables[1:])
    case = Case(
        "case-mut-normalized-string-membership",
        1,
        tables,
        Program("prog-mut-normalized-string-membership", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_sql_distinct_null_topk_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("s", "str"), ColumnSpec("flag", "bool")],
        [
            {"id": 0, "g": "a", "s": "", "flag": True},
            {"id": 1, "g": None, "s": None, "flag": None},
            {"id": 2, "g": "space value", "s": " Alpha ", "flag": False},
        ],
    )
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_sql_distinct_null_topk([table], operations, random.Random(1))

    assert detail.startswith("append_sql_distinct_null_topk:")
    assert [op["op"] for op in operations[-4:]] == ["distinct", "sort", "offset", "limit"]
    assert any(op["op"] == "coalesce" for op in operations)
    case = Case("case-mut-sql-distinct", 1, [table], Program("prog-mut-sql-distinct", 1, operations))
    assert validate_case_program(case) == []


def test_append_left_join_coalesce_membership_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("row_nr", "int"), ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
        [
            {"row_nr": 0, "id": 0, "g": "a", "x": 1},
            {"row_nr": 1, "id": 1, "g": None, "x": 2},
            {"row_nr": 2, "id": 2, "g": "space value", "x": 3},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_left_join_coalesce_membership(tables, operations, random.Random(1))

    assert detail.startswith("append_left_join_coalesce_membership:")
    assert any(op["op"] == "join" and op.get("how") == "left" for op in operations)
    assert any(op["op"] == "coalesce" for op in operations)
    assert any(op["op"] in {"semi_join", "anti_join"} for op in operations)
    assert len(tables) == 3
    case = Case(
        "case-mut-left-join-coalesce-membership",
        1,
        tables,
        Program("prog-mut-left-join-coalesce-membership", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_left_join_case_membership_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("row_nr", "int"), ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("x", "int")],
        [
            {"row_nr": 0, "id": 0, "g": "a", "x": 1},
            {"row_nr": 1, "id": 1, "g": None, "x": 2},
            {"row_nr": 2, "id": 2, "g": "space value", "x": 3},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_left_join_case_membership(tables, operations, random.Random(1))

    assert detail.startswith("append_left_join_case_membership:")
    assert any(op["op"] == "join" and op.get("how") == "left" for op in operations)
    assert any(op["op"] == "case_when" for op in operations)
    assert any(op["op"] == "groupby" for op in operations)
    assert len(tables) == 2
    case = Case(
        "case-mut-left-join-case-membership",
        1,
        tables,
        Program("prog-mut-left-join-case-membership", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_empty_filter_global_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int"),
            ColumnSpec("y", "float"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1, "y": 0.5, "flag": True},
            {"id": 1, "x": None, "y": None, "flag": None},
            {"id": 2, "x": -1, "y": -0.5, "flag": False},
        ],
    )
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_empty_filter_global_aggregate([table], operations, random.Random(1))

    assert detail.startswith("append_empty_filter_global_aggregate:")
    assert [op["op"] for op in operations[-2:]] == ["filter", "aggregate"]
    assert any(agg["func"] in {"any", "all"} for agg in operations[-1]["aggs"])
    case = Case("case-mut-empty-global-aggregate", 1, [table], Program("prog-mut-empty-global-aggregate", 1, operations))
    assert validate_case_program(case) == []


def test_append_sql_union_coalesce_distinct_topk_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("g", "str"), ColumnSpec("s", "str"), ColumnSpec("flag", "bool")],
        [
            {"id": 0, "g": "a", "s": "", "flag": True},
            {"id": 1, "g": None, "s": None, "flag": None},
            {"id": 2, "g": "space value", "s": " Alpha ", "flag": False},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_sql_union_coalesce_distinct_topk(tables, operations, random.Random(1))

    assert detail.startswith("append_sql_union_coalesce_distinct_topk:")
    assert operations[-5]["op"] in {"coalesce", "mutate"}
    assert [op["op"] for op in operations[-4:]] == ["distinct", "sort", "offset", "limit"]
    assert any(op["op"] == "union_all" for op in operations)
    assert len(tables) == 2
    case = Case("case-mut-sql-union-distinct", 1, tables, Program("prog-mut-sql-union-distinct", 1, operations))
    assert validate_case_program(case) == []


def test_append_boolean_membership_case_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1, "flag": True},
            {"id": 1, "x": 2, "flag": None},
            {"id": 2, "x": 3, "flag": False},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_boolean_membership_case_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_boolean_membership_case_aggregate:")
    assert [op["op"] for op in operations[-5:]] == ["filter", "semi_join", "case_when", "groupby", "sort"]
    assert operations[-3]["condition"]["cmp"] == "bool_is_true"
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-bool-membership-case-agg",
        1,
        tables,
        Program("prog-mut-bool-membership-case-agg", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_left_join_boolean_case_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1, "flag": True},
            {"id": 1, "x": None, "flag": None},
            {"id": 2, "x": 3, "flag": False},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_left_join_boolean_case_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_left_join_boolean_case_aggregate:")
    assert [op["op"] for op in operations[-4:]] == ["join", "case_when", "groupby", "sort"]
    assert operations[-4]["how"] == "left"
    assert operations[-3]["then"] is True
    assert operations[-3]["else"] is False
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-left-join-bool-case-agg",
        1,
        tables,
        Program("prog-mut-left-join-bool-case-agg", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_left_join_boolean_coalesce_case_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1, "flag": True},
            {"id": 1, "x": None, "flag": None},
            {"id": 2, "x": 3, "flag": False},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_left_join_boolean_coalesce_case_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_left_join_boolean_coalesce_case_aggregate:")
    assert [op["op"] for op in operations[-5:]] == ["join", "coalesce", "case_when", "groupby", "sort"]
    assert operations[-5]["how"] == "left"
    assert operations[-4]["fallback"] is False
    assert operations[-3]["condition"]["cmp"] == "bool_is_true"
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-left-join-bool-coalesce-case-agg",
        1,
        tables,
        Program("prog-mut-left-join-bool-coalesce-case-agg", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_boolean_antijoin_case_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1, "flag": True},
            {"id": 1, "x": 2, "flag": None},
            {"id": 2, "x": 3, "flag": False},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_boolean_antijoin_case_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_boolean_antijoin_case_aggregate:")
    assert [op["op"] for op in operations[-5:]] == ["filter", "anti_join", "case_when", "groupby", "sort"]
    assert operations[-3]["condition"]["cmp"] == "bool_is_true"
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-bool-antijoin-case-agg",
        1,
        tables,
        Program("prog-mut-bool-antijoin-case-agg", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_left_join_boolean_coalesce_filter_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "x": 1, "flag": True},
            {"id": 1, "x": None, "flag": None},
            {"id": 2, "x": 3, "flag": False},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_left_join_boolean_coalesce_filter_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_left_join_boolean_coalesce_filter_aggregate:")
    assert [op["op"] for op in operations[-6:]] == ["join", "coalesce", "filter", "case_when", "groupby", "sort"]
    assert operations[-6]["how"] == "left"
    assert operations[-5]["fallback"] is False
    assert operations[-4]["cmp"] == "=="
    assert isinstance(operations[-4]["value"], bool)
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-left-join-bool-coalesce-filter-agg",
        1,
        tables,
        Program("prog-mut-left-join-bool-coalesce-filter-agg", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_numeric_text_boolean_antijoin_case_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("num_s", "str"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "num_s": "0", "flag": True},
            {"id": 1, "num_s": "1", "flag": None},
            {"id": 2, "num_s": "2", "flag": False},
            {"id": 3, "num_s": "5", "flag": True},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_numeric_text_boolean_antijoin_case_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_numeric_text_boolean_antijoin_case_aggregate:")
    assert [op["op"] for op in operations[-7:]] == [
        "mutate",
        "filter",
        "filter",
        "anti_join",
        "case_when",
        "groupby",
        "sort",
    ]
    assert operations[-7]["expr"]["input_domain"] == "integer_string"
    assert operations[-5]["cmp"] == "bool_is_not_false"
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-numeric-text-bool-antijoin-case-agg",
        1,
        tables,
        Program("prog-mut-numeric-text-bool-antijoin-case-agg", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_multi_key_membership_case_aggregate_mutation_stays_valid():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("g", "str"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
        ],
        [
            {"id": 0, "g": "a", "x": 1, "flag": True},
            {"id": 1, "g": "b", "x": 2, "flag": None},
            {"id": 2, "g": "c", "x": 3, "flag": False},
            {"id": 3, "g": None, "x": 5, "flag": True},
        ],
    )
    tables = [table]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_multi_key_membership_case_aggregate(tables, operations, random.Random(1))

    assert detail.startswith("append_multi_key_membership_case_aggregate:")
    assert [op["op"] for op in operations[-5:]] == ["filter", "semi_join", "case_when", "groupby", "sort"]
    assert isinstance(operations[-4]["left_on"], list)
    assert operations[-4]["left_on"] == operations[-4]["right_on"]
    assert operations[-3]["condition"]["cmp"] == "bool_is_true"
    assert any(agg["func"] in {"any", "all"} for agg in operations[-2]["aggs"])
    assert len(tables) == 2
    case = Case(
        "case-mut-multi-key-membership-case-agg",
        1,
        tables,
        Program("prog-mut-multi-key-membership-case-agg", 1, operations),
    )
    assert validate_case_program(case) == []
