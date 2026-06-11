import random

import datadiff.mutator as mutator_module
import datadiff.mutator_ir.rewrite_groupby as rewrite_groupby_module
import datadiff.mutator_ir.rewrite_splice as rewrite_splice_module
import datadiff.mutator_ir.rewrite_swap as rewrite_swap_module
from datadiff.classification_oracle import validate_case_program
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.mutator import (
    DISCOVERY_MUTATION_OPERATOR_NAMES,
    MUTATION_OPERATOR_NAMES,
    PROBE_MUTATION_OPERATOR_NAMES,
    ROOT_TARGETED_MUTATION_OPERATOR_NAMES,
    SHRINK_MUTATION_OPERATOR_NAMES,
    SPECIALIZED_DISCOVERY_MUTATION_OPERATOR_NAMES,
    TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES,
    TABLE_MUTATING_MUTATION_OPERATOR_NAMES,
    TABLE_ONLY_MUTATION_OPERATOR_NAMES,
    mutation_operator_profiles,
    _append_group_quantile_probe,
    _append_scalar_subquery_probe,
    _append_window_avg_probe,
    _append_struct_distinct_probe,
    _append_bit_compare_probe,
    _append_round_even_probe,
    _append_float_literal_precision_probe,
    _append_timestamp_precision_filter_probe,
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
    _append_coalesce_sort_topk,
    _append_boolean_membership_case_aggregate,
    _append_left_join_boolean_case_aggregate,
    _append_left_join_boolean_coalesce_case_aggregate,
    _append_boolean_antijoin_case_aggregate,
    _append_left_join_boolean_coalesce_filter_aggregate,
    _append_numeric_text_boolean_antijoin_case_aggregate,
    _append_multi_key_membership_case_aggregate,
    _append_join_filter_groupby_topk,
    _append_index_bool_probe,
    _append_empty_literal_groupby_probe,
    _append_arrow_string_eq_sum_probe,
    _append_arrow_timestamp_loc_slice_probe,
    _append_arrow_timestamp_index_attr_probe,
    _append_eval_inplace_alias_probe,
    _append_arrow_bool_groupby_reduction_probe,
    _append_dataset_isin_all_match_probe,
    _append_large_string_partition_probe,
    _append_hash_pivot_wider_probe,
    _append_list_flatten_parent_indices_probe,
    _append_rolling_mean_by_null_count_probe,
    _append_csv_long_numeric_roundtrip_probe,
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
from datadiff.mutator_ir import (
    apply_adjacent_independent_swap,
    apply_filter_above_groupby,
    apply_filter_pushdown,
    apply_redundant_op_fold,
    apply_subtree_splice,
    apply_wrap_with_window,
    legal_adjacent_swap_positions,
    legal_filter_above_groupby_positions,
    legal_filter_pushdown_positions,
    legal_redundant_fold_positions,
    legal_subtree_splice_positions,
    legal_window_wrap_positions,
)
from datadiff.mutator_shrink import (
    shrink_drop_tail_op,
    shrink_fold_redundant_op,
    shrink_inline_single_use_mutate,
    shrink_merge_adjacent_filters,
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


def test_random_operation_can_append_bool_not_for_bool_only_available_columns():
    table = TableData(
        "t0",
        [ColumnSpec("flag", "bool")],
        [{"flag": True}, {"flag": None}, {"flag": False}],
    )

    seen = []
    for seed in range(800):
        op = _random_operation([table], [], random.Random(seed))
        if op is not None and op["op"] == "mutate":
            seen.append(op["expr"])

    assert any(expr["kind"] == "bool_not" and expr["source"] == "flag" for expr in seen)


def test_random_operation_can_append_numeric_string_cast_targets():
    table = TableData(
        "t0",
        [ColumnSpec("num_s", "str")],
        [{"num_s": "1"}, {"num_s": "-2"}, {"num_s": None}, {"num_s": "10"}],
    )

    cast_targets = set()
    for seed in range(2400):
        op = _random_operation([table], [], random.Random(seed))
        if op is None or op["op"] != "mutate":
            continue
        expr = op["expr"]
        if expr["kind"] == "cast" and expr.get("input_domain") == "integer_string":
            cast_targets.add(expr["to"])

    assert cast_targets == {"int", "float"}


def test_random_operation_covers_high_risk_structural_ops():
    table = TableData(
        "t0",
        [
            ColumnSpec("row_nr", "int"),
            ColumnSpec("id", "int"),
            ColumnSpec("g", "str"),
            ColumnSpec("x", "int"),
            ColumnSpec("flag", "bool"),
            ColumnSpec("s", "str"),
        ],
        [
            {"row_nr": 0, "id": 0, "g": "a", "x": 1, "flag": True, "s": "Alpha"},
            {"row_nr": 1, "id": 0, "g": " b ", "x": None, "flag": None, "s": ""},
            {"row_nr": 2, "id": 1, "g": None, "x": -2, "flag": False, "s": "space value"},
            {"row_nr": 3, "id": 1, "g": "space value", "x": 5, "flag": True, "s": "Beta"},
        ],
    )

    target_ops = {"coalesce", "case_when", "row_number_filter", "running_sum", "fill_null", "distinct", "offset"}
    seen_ops = set()

    for seed in range(4000):
        op = _random_operation([table], [], random.Random(seed))
        if op is None or op["op"] not in target_ops:
            continue
        seen_ops.add(op["op"])
        case = Case(f"case-random-op-{seed}", seed, [table], Program(f"prog-random-op-{seed}", seed, [op]))
        assert validate_case_program(case) == []
        if seen_ops == target_ops:
            break

    assert seen_ops == target_ops


def test_random_operation_composes_derived_null_and_window_paths():
    table = TableData(
        "t0",
        [
            ColumnSpec("row_nr", "int"),
            ColumnSpec("id", "int"),
            ColumnSpec("g", "str"),
            ColumnSpec("x", "int"),
            ColumnSpec("y", "float"),
            ColumnSpec("flag", "bool"),
            ColumnSpec("s", "str"),
        ],
        [
            {"row_nr": 0, "id": 0, "g": "a", "x": 1, "y": 0.5, "flag": True, "s": "Alpha"},
            {"row_nr": 1, "id": 0, "g": None, "x": None, "y": None, "flag": None, "s": ""},
            {"row_nr": 2, "id": 1, "g": "space value", "x": -2, "y": -1.5, "flag": False, "s": "Beta"},
        ],
    )
    prefix_ops = [
        {"op": "mutate", "column": "m_x", "expr": {"kind": "add_const", "source": "x", "value": 1}},
        {"op": "coalesce", "columns": ["g", "s"], "as": "g_or_s", "fallback": "missing"},
        {
            "op": "case_when",
            "as": "flag_bucket",
            "condition": {"column": "flag", "cmp": "is_null", "value": None},
            "then": "unknown",
            "else": "known",
        },
    ]

    seen = {
        "filter_on_derived": False,
        "case_when_on_derived": False,
        "case_when_null_cmp": False,
        "row_number_order_derived": False,
        "running_sum_on_derived": False,
    }

    for seed in range(3000):
        op = _random_operation([table], prefix_ops, random.Random(seed))
        if op is None:
            continue
        case = Case(
            f"case-random-composite-{seed}",
            seed,
            [table],
            Program(f"prog-random-composite-{seed}", seed, [*prefix_ops, op]),
        )
        assert validate_case_program(case) == []
        if op["op"] == "filter" and op.get("column") in {"m_x", "g_or_s", "flag_bucket"}:
            seen["filter_on_derived"] = True
        elif op["op"] == "case_when" and op.get("condition", {}).get("column") in {"m_x", "g_or_s", "flag_bucket"}:
            seen["case_when_on_derived"] = True
        elif op["op"] == "case_when" and op.get("condition", {}).get("cmp") in {"is_null", "is_not_null"}:
            seen["case_when_null_cmp"] = True
        elif op["op"] == "row_number_filter" and any(
            key["column"] in {"m_x", "g_or_s", "flag_bucket"} for key in op.get("order_by", [])
        ):
            seen["row_number_order_derived"] = True
        elif op["op"] == "running_sum" and op.get("source") == "m_x":
            seen["running_sum_on_derived"] = True
        if all(seen.values()):
            break

    assert all(seen.values()), seen


def test_mutation_operator_profiles_expose_semantic_family_affinity_alias():
    profile = mutation_operator_profiles(allow_probe_operators=False)["append_left_join_case_membership"]

    assert "conditional_semantics" in profile.semantic_family_affinity
    assert profile.semantic_affinity == profile.semantic_family_affinity
    assert "cross_model_consistency" in profile.exploration_objective_affinity


def test_mutation_operator_profiles_can_filter_shrink_mutations():
    profiles = mutation_operator_profiles(
        allow_probe_operators=False,
        enable_shrink_mutations=False,
    )

    assert SHRINK_MUTATION_OPERATOR_NAMES
    assert set(profiles).isdisjoint(SHRINK_MUTATION_OPERATOR_NAMES)


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


def test_mutation_state_cache_reuses_schema_and_context(monkeypatch):
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("g", "str")],
        [{"x": 1, "g": "Alpha"}, {"x": None, "g": None}],
    )
    operations = [
        {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "x", "value": 1}},
        {"op": "select", "columns": ["g", "m_1"]},
    ]

    original = mutator_module.state_after_operations
    calls = {"count": 0}

    def counting_state_after_operations(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mutator_module, "state_after_operations", counting_state_after_operations)

    mutator_module._available_columns([table], operations)
    mutator_module._available_columns([table], operations)
    assert calls["count"] == 2

    calls["count"] = 0
    cache = mutator_module._MutationStateCache()
    with mutator_module._activate_mutation_state_cache(cache):
        first = mutator_module._available_columns([table], operations)
        second = mutator_module._available_columns([table], operations)
        context = mutator_module._mutation_operation_context([table], operations)

    assert first == ["g", "m_1"]
    assert second == first
    assert context is not None
    assert context.available == ("g", "m_1")
    assert calls["count"] == 1


def test_mutation_state_cache_preserves_schema_results():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("g", "str")],
        [{"x": 1, "g": "Alpha"}, {"x": None, "g": None}],
    )
    operations = [
        {"op": "mutate", "column": "m_1", "expr": {"kind": "add_const", "source": "x", "value": 1}},
        {
            "op": "case_when",
            "as": "bucket",
            "condition": {"column": "g", "cmp": "is_null", "value": None},
            "then": "missing",
            "else": "present",
        },
        {"op": "select", "columns": ["m_1", "bucket"]},
    ]

    uncached = mutator_module._mutation_schema([table], operations)
    with mutator_module._activate_mutation_state_cache(mutator_module._MutationStateCache()):
        cached = mutator_module._mutation_schema([table], operations)
        cached_again = mutator_module._mutation_schema([table], operations)

    assert cached.columns == uncached.columns
    assert cached.column_types == uncached.column_types
    assert cached.nullable_columns == uncached.nullable_columns
    assert cached_again.columns == uncached.columns
    assert cached_again.column_types == uncached.column_types
    assert cached_again.nullable_columns == uncached.nullable_columns


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


def test_mutate_case_with_metadata_records_value_catalog_entries(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": None}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )
    monkeypatch.setattr(mutator_module, "VALUE_CATALOG_SAMPLE_PROBABILITY", 1.0)
    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (mutator_module.MutationOperator("value", mutator_module._mutate_scalar_value),),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        value_catalog_scores={"int.zero": 6.0},
    )

    entries = result.metadata["mutation"]["value_catalog_entries"]
    assert entries
    assert entries[0]["entry_id"].startswith("int.")
    assert entries[0]["column"] == "x"
    assert entries[0]["source_root_cause"]
    assert result.metadata["mutation"]["detail"].startswith("value_catalog:")


def test_mutate_case_can_disable_value_catalog_sampling(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": None}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )
    monkeypatch.setattr(mutator_module, "VALUE_CATALOG_SAMPLE_PROBABILITY", 1.0)
    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (mutator_module.MutationOperator("value", mutator_module._mutate_scalar_value),),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        enable_value_catalog=False,
        value_catalog_scores={"int.zero": 6.0},
    )

    assert result.metadata["mutation"]["value_catalog_entries"] == []
    assert result.metadata["mutation"]["detail"].startswith("value:int:")


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


def test_unproductive_unchanged_simulation_skips_repair(monkeypatch):
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    operations = [{"op": "limit", "n": 1}]

    def fail(tables, operations, rnd):
        del tables, operations, rnd
        return "fail:no-compatible-column"

    def fail_repair(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unproductive unchanged candidates should not be repaired")

    monkeypatch.setattr(mutator_module, "repair_operations", fail_repair)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator("fail", fail),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_tables == [table]
    assert trial_operations == operations
    assert step.changed is False
    assert step.productive is False


def test_table_only_simulation_skips_operation_clone_and_repair(monkeypatch):
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    operations = [{"op": "limit", "n": 1}]

    def fail_repair(*args, **kwargs):
        del args, kwargs
        raise AssertionError("table-only candidates should not repair unchanged operations")

    monkeypatch.setattr(mutator_module, "repair_operations", fail_repair)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator("duplicate_row", mutator_module._duplicate_row),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_operations is operations
    assert len(trial_tables[0].rows) == 2
    assert len(table.rows) == 1
    assert step.changed is True
    assert step.productive is True


def test_table_only_simulation_shallow_copies_extra_tables(monkeypatch):
    primary = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    extra = TableData("t1", [ColumnSpec("k", "int")], [{"k": 1}])
    operations = [{"op": "limit", "n": 1}]
    clone_table = mutator_module._clone_table

    def clone_primary_only(table_arg):
        if table_arg is extra:
            raise AssertionError("table-only simulation should not deep-clone extra tables")
        return clone_table(table_arg)

    monkeypatch.setattr(mutator_module, "_clone_table", clone_primary_only)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [primary, extra],
        operations,
        mutator_module.MutationOperator("duplicate_row", mutator_module._duplicate_row),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_tables[0] is not primary
    assert trial_tables[1] is extra
    assert len(trial_tables[0].rows) == 2
    assert len(primary.rows) == 1
    assert trial_operations is operations
    assert step.changed is True
    assert step.productive is True


def test_mutation_state_table_cache_can_be_invalidated_after_in_place_table_mutation():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    cache = mutator_module._MutationStateCache()

    before = mutator_module._mutation_state_cache_key([table], [], cache=cache)
    table.rows[0]["x"] = None
    cached = mutator_module._mutation_state_cache_key([table], [], cache=cache)
    mutator_module._invalidate_table_cache(cache, [table])
    after = mutator_module._mutation_state_cache_key([table], [], cache=cache)

    assert cached == before
    assert after != before


def test_mutation_state_operation_cache_can_be_invalidated_after_in_place_operation_mutation():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    operation = {"op": "limit", "n": 1}
    cache = mutator_module._MutationStateCache()

    before = mutator_module._mutation_state_cache_key([table], [operation], cache=cache)
    operation["n"] = 2
    cached = mutator_module._mutation_state_cache_key([table], [operation], cache=cache)
    mutator_module._invalidate_operation_cache(cache, [operation])
    after = mutator_module._mutation_state_cache_key([table], [operation], cache=cache)

    assert cached == before
    assert after != before


def test_append_only_simulation_shallow_copies_operation_list(monkeypatch):
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    operations = [{"op": "limit", "n": 1}]

    def append_only(tables, operations_arg, rnd):
        del tables, rnd
        operations_arg.append({"op": "limit", "n": 2})
        return "append:limit"

    def fail_clone_operation(operation):
        del operation
        raise AssertionError("append-only simulation should not deep-clone existing operations")

    def passthrough_repair(table_arg, operations_arg, extra_tables=None):
        del table_arg, extra_tables
        return list(operations_arg)

    monkeypatch.setattr(mutator_module, "_clone_operation", fail_clone_operation)
    monkeypatch.setattr(mutator_module, "repair_operations", passthrough_repair)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator("append_op", append_only),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_tables is not None
    assert trial_operations is not operations
    assert trial_operations[0] is operations[0]
    assert trial_operations[1] == {"op": "limit", "n": 2}
    assert operations == [{"op": "limit", "n": 1}]
    assert step.changed is True
    assert step.productive is True


def test_operation_list_only_simulation_shallow_copies_operation_list(monkeypatch):
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    operations = [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "keys": [{"column": "y", "ascending": True, "nulls": "last"}]},
        {"op": "limit", "n": 2},
    ]

    def fail_clone_operation(operation):
        del operation
        raise AssertionError("operation-list-only simulation should not deep-clone existing operations")

    def passthrough_repair(table_arg, operations_arg, extra_tables=None):
        del table_arg, extra_tables
        return list(operations_arg)

    monkeypatch.setattr(mutator_module, "_clone_operation", fail_clone_operation)
    monkeypatch.setattr(mutator_module, "repair_operations", passthrough_repair)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator("ir_swap_adjacent", apply_adjacent_independent_swap),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_operations is not operations
    assert trial_operations != operations
    assert sorted(id(operation) for operation in trial_operations) == sorted(id(operation) for operation in operations)
    assert operations == [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "keys": [{"column": "y", "ascending": True, "nulls": "last"}]},
        {"op": "limit", "n": 2},
    ]
    assert trial_tables[0] is table
    assert step.changed is True
    assert step.productive is True


def test_append_table_only_simulation_shallow_copies_table_list(monkeypatch):
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("label", "str", nullable=True),
        ],
        [
            {"id": 1, "label": "A"},
            {"id": 2, "label": "B"},
        ],
    )
    operations = [{"op": "limit", "n": 2}]

    def fail_clone_table(table_arg):
        del table_arg
        raise AssertionError("append-table-only simulation should not deep-clone existing tables")

    def passthrough_repair(table_arg, operations_arg, extra_tables=None):
        del table_arg, extra_tables
        return list(operations_arg)

    monkeypatch.setattr(mutator_module, "_clone_table", fail_clone_table)
    monkeypatch.setattr(mutator_module, "repair_operations", passthrough_repair)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator(
            "append_left_join_case_membership",
            mutator_module._append_left_join_case_membership,
        ),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_tables[0] is table
    assert len(trial_tables) == 2
    assert len(table.rows) == 2
    assert len(trial_operations) > len(operations)
    assert operations == [{"op": "limit", "n": 2}]
    assert step.changed is True
    assert step.productive is True


def test_numeric_text_append_table_simulation_keeps_conservative_table_clone(monkeypatch):
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("flag", "bool", nullable=True),
        ],
        [
            {"id": 1, "flag": True},
            {"id": 2, "flag": False},
        ],
    )
    operations = [{"op": "limit", "n": 2}]

    def passthrough_repair(table_arg, operations_arg, extra_tables=None):
        del table_arg, extra_tables
        return list(operations_arg)

    monkeypatch.setattr(mutator_module, "repair_operations", passthrough_repair)

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator(
            "append_numeric_text_boolean_antijoin_case_aggregate",
            mutator_module._append_numeric_text_boolean_antijoin_case_aggregate,
        ),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_tables[0] is not table
    assert len(trial_tables[0].columns) == 3
    assert len(table.columns) == 2
    assert all("id_num_s" not in row for row in table.rows)
    assert trial_operations != operations
    assert step.changed is True
    assert step.productive is True


def test_unknown_simulation_operator_uses_conservative_table_clone():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])
    operations = [{"op": "limit", "n": 1}]

    def custom(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = 9
        return "custom:x"

    trial_tables, trial_operations, step = mutator_module._simulate_mutation_step(
        [table],
        operations,
        mutator_module.MutationOperator("custom_unknown", custom),
        seed=99,
        case_seed=10,
        step_index=0,
        candidate_rank=1,
        candidate_width=1,
        operator_scores={},
        target_context={"semantic_family": set(), "semantic_signal": set(), "exploration_objective": set()},
        disagreement=None,
        enable_divergence_conditioned_mutations=True,
        enable_value_catalog=False,
        value_catalog_scores=None,
        prior_steps=[],
        state_cache=mutator_module._MutationStateCache(),
    )

    assert trial_tables[0].rows[0]["x"] == 9
    assert table.rows[0]["x"] == 1
    assert trial_operations == operations
    assert trial_operations is not operations
    assert step.changed is True


def test_clone_operation_unwraps_irnode_to_independent_dict():
    program = Program(
        "prog-sort",
        1,
        [
            {
                "op": "sort",
                "keys": [{"column": "x", "ascending": True, "nulls": "last"}],
            }
        ],
    )
    operation = program.operations[0]

    cloned = mutator_module._clone_operation(operation)

    assert isinstance(operation, mutator_module.IRNode)
    assert isinstance(cloned, dict)
    assert not isinstance(cloned["keys"][0], mutator_module.IRNode)
    cloned["keys"][0]["ascending"] = False
    assert operation["keys"][0]["ascending"] is True


def test_clone_row_keeps_mutable_values_independent():
    row = {"x": 1, "payload": {"values": [1, 2]}}

    cloned = mutator_module._clone_row(row)

    assert cloned == row
    assert cloned is not row
    cloned["payload"]["values"].append(3)
    assert row["payload"]["values"] == [1, 2]


def test_mutation_operation_cache_key_normalizes_nested_values():
    key = mutator_module._mutation_operation_cache_key(
        {
            "op": "filter",
            "value": float("nan"),
            "bounds": [float("-inf"), 3],
            "tags": {"b", "a"},
        }
    )

    assert ("value", ("float", "nan")) in key
    assert ("bounds", (("float", "inf", -1), 3)) in key
    assert ("tags", ("a", "b")) in key


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


def test_mutate_case_prioritizes_divergence_affine_operator(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    def neutral(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = -1
        return "neutral:x"

    def numeric(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = 99
        return "numeric:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("neutral", neutral),
            mutator_module.MutationOperator(
                "numeric",
                numeric,
                divergence_affinity=("disagree_class:numeric",),
            ),
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        disagreement={
            "column_classes": {"x": "numeric"},
            "mismatch_class": "value",
        },
    )

    assert result.metadata["mutation"]["operator"] == "numeric"
    assert result.case.tables[0].rows[0]["x"] == 99
    assert result.metadata["mutation_plan"]["steps"][0]["divergence_affinity"] > 0.0


def test_mutate_case_can_ablate_divergence_conditioned_affinity(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    def numeric(tables, operations, rnd):
        del operations, rnd
        tables[0].rows[0]["x"] = 99
        return "numeric:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator(
                "numeric",
                numeric,
                divergence_affinity=("disagree_class:numeric",),
            ),
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        disagreement={
            "column_classes": {"x": "numeric"},
            "mismatch_class": "value",
        },
        enable_divergence_conditioned_mutations=False,
    )

    assert result.metadata["mutation"]["operator"] == "numeric"
    assert result.metadata["mutation_plan"]["steps"][0]["divergence_affinity"] == 0.0


def test_mutate_case_uses_operator_energy_for_candidate_width(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    call_counts = {"cold": 0, "hot": 0}

    def cold(tables, operations, rnd):
        del operations, rnd
        call_counts["cold"] += 1
        tables[0].rows[0]["x"] = 10 + call_counts["cold"]
        return f"cold:{call_counts['cold']}"

    def hot(tables, operations, rnd):
        del operations, rnd
        call_counts["hot"] += 1
        tables[0].rows[0]["x"] = -10
        return "hot:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("cold", cold),
            mutator_module.MutationOperator("hot", hot),
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"cold": 3.0, "hot": 0.0},
        operator_pulls={"hot": 20},
        recent_operator_pulls={"hot": 8},
    )

    assert result.metadata["mutation"]["operator"] == "cold"
    assert call_counts["cold"] > 1
    assert result.metadata["mutation_plan"]["steps"][0]["candidate_width"] > 1


def test_mutate_case_can_disable_operator_energy_candidate_width(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )

    call_counts = {"cold": 0, "hot": 0}

    def cold(tables, operations, rnd):
        del operations, rnd
        call_counts["cold"] += 1
        tables[0].rows[0]["x"] = 10 + call_counts["cold"]
        return f"cold:{call_counts['cold']}"

    def hot(tables, operations, rnd):
        del operations, rnd
        call_counts["hot"] += 1
        tables[0].rows[0]["x"] = -10
        return "hot:x"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("cold", cold),
            mutator_module.MutationOperator("hot", hot),
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"cold": 3.0, "hot": 0.0},
        operator_pulls={"hot": 20},
        recent_operator_pulls={"hot": 8},
        enable_per_operator_energy=False,
    )

    assert result.metadata["mutation"]["operator"] == "cold"
    assert result.metadata["mutation_plan"]["steps"][0]["candidate_width"] == 1


def test_mutate_case_can_disable_ir_rewrite_mutations(monkeypatch):
    base = Case(
        "case-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}])],
        Program("prog-base", 10, [{"op": "limit", "n": 1}]),
    )
    calls = {"ir": 0, "plain": 0}

    def ir_rewrite(tables, operations, rnd):
        del tables, operations, rnd
        calls["ir"] += 1
        return "ir_swap_adjacent:test"

    def plain(tables, operations, rnd):
        del operations, rnd
        calls["plain"] += 1
        tables[0].rows[0]["x"] = 2
        return "plain:value"

    monkeypatch.setattr(
        mutator_module,
        "MUTATION_OPERATORS",
        (
            mutator_module.MutationOperator("ir_swap_adjacent", ir_rewrite),
            mutator_module.MutationOperator("plain", plain),
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_swap_adjacent": 10.0, "plain": 1.0, "__untried__": -10.0},
        enable_ir_rewrite_mutations=False,
    )

    assert result.metadata["mutation"]["operator"] == "plain"
    assert calls["ir"] == 0
    assert calls["plain"] > 0
    profiles = mutation_operator_profiles(
        allow_probe_operators=False,
        enable_ir_rewrite_mutations=False,
    )
    assert "ir_swap_adjacent" not in profiles
    assert "ir_pushdown_filter" not in profiles
    assert "ir_pull_filter_above_groupby" not in profiles
    assert "ir_wrap_with_window" not in profiles
    assert "ir_splice_subtree" not in profiles
    assert "ir_fold_redundant_op" not in profiles


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


def test_ir_adjacent_swap_reorders_independent_local_operations_and_preserves_validity():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    operations = [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "keys": [{"column": "y", "ascending": True, "nulls": "last"}]},
        {"op": "limit", "n": 2},
    ]

    assert legal_adjacent_swap_positions([table], operations) == [0, 1]
    detail = apply_adjacent_independent_swap([table], operations, random.Random(1))

    assert detail.startswith("ir_swap_adjacent:")
    assert operations != [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "keys": [{"column": "y", "ascending": True, "nulls": "last"}]},
        {"op": "limit", "n": 2},
    ]
    case = Case("case-ir-swap", 1, [table], Program("prog-ir-swap", 1, operations))
    assert validate_case_program(case) == []


def test_legal_adjacent_swap_positions_reuses_prefix_states(monkeypatch):
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    operations = [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "keys": [{"column": "y", "ascending": True, "nulls": "last"}]},
        {"op": "limit", "n": 2},
    ]

    def fail_state_after_operations(*args, **kwargs):
        del args, kwargs
        raise AssertionError("swap position enumeration should reuse precomputed prefix states")

    monkeypatch.setattr(rewrite_swap_module, "state_after_operations", fail_state_after_operations)

    assert rewrite_swap_module.legal_adjacent_swap_positions([table], operations) == [0, 1]


def test_ir_adjacent_swap_respects_barriers_and_dependencies():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
    )
    grouped = [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "groupby", "keys": ["x"], "aggs": [{"column": "y", "func": "sum", "as": "sum_y"}]},
        {"op": "filter", "column": "sum_y", "cmp": ">=", "value": 0},
    ]

    assert legal_adjacent_swap_positions([table], grouped) == []
    assert apply_adjacent_independent_swap([table], grouped, random.Random(1)) == "ir_swap_adjacent:no-legal-position"


def test_mutate_case_can_select_ir_adjacent_swap_operator():
    base = Case(
        "case-ir-swap-base",
        10,
        [TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("y", "int")], [{"x": 1, "y": 2}, {"x": 3, "y": 4}])],
        Program(
            "prog-ir-swap-base",
            10,
            [
                {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
                {"op": "sort", "keys": [{"column": "y", "ascending": True, "nulls": "last"}]},
                {"op": "limit", "n": 2},
            ],
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_swap_adjacent": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )

    assert result.metadata["mutation"]["operator"] == "ir_swap_adjacent"
    assert result.metadata["mutation"]["detail"].startswith("ir_swap_adjacent:")
    assert result.case.program.operations != base.program.operations
    assert validate_case_program(result.case) == []


def test_ir_filter_pushdown_moves_filter_before_join_when_left_columns_are_available():
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
    operations = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
        {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
        {"op": "sort", "keys": [{"column": "tag", "ascending": True, "nulls": "last"}]},
    ]

    assert legal_filter_pushdown_positions([left, right], operations) == [1]
    detail = apply_filter_pushdown([left, right], operations, random.Random(1))

    assert detail == "ir_pushdown_filter:1:join:filter[x]"
    assert operations[:2] == [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
    ]
    case = Case("case-ir-pushdown", 1, [left, right], Program("prog-ir-pushdown", 1, operations))
    assert validate_case_program(case) == []


def test_ir_filter_pushdown_respects_join_and_mutate_dependencies():
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
    right_column_filter = [
        {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
        {"op": "filter", "column": "tag", "cmp": "==", "value": "a"},
    ]
    derived_column_filter = [
        {"op": "mutate", "column": "m_x", "expr": {"kind": "add_const", "source": "x", "value": 1}},
        {"op": "filter", "column": "m_x", "cmp": ">=", "value": 0},
    ]

    assert legal_filter_pushdown_positions([left, right], right_column_filter) == []
    assert apply_filter_pushdown([left, right], right_column_filter, random.Random(1)) == (
        "ir_pushdown_filter:no-legal-position"
    )
    assert legal_filter_pushdown_positions([left], derived_column_filter) == []
    assert apply_filter_pushdown([left], derived_column_filter, random.Random(1)) == (
        "ir_pushdown_filter:no-legal-position"
    )


def test_mutate_case_can_select_ir_filter_pushdown_operator():
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
        "case-ir-pushdown-base",
        10,
        [left, right],
        Program(
            "prog-ir-pushdown-base",
            10,
            [
                {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "inner"},
                {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
                {"op": "sort", "keys": [{"column": "tag", "ascending": True, "nulls": "last"}]},
            ],
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_pushdown_filter": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )

    assert result.metadata["mutation"]["operator"] == "ir_pushdown_filter"
    assert result.metadata["mutation"]["detail"].startswith("ir_pushdown_filter:")
    assert result.case.program.operations != base.program.operations
    assert validate_case_program(result.case) == []


def test_ir_pull_filter_above_groupby_rewrites_having_like_filter_to_source_column():
    table = TableData(
        "t0",
        [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
        [{"g": "a", "x": 1}, {"g": "a", "x": 2}, {"g": "b", "x": 5}],
    )
    operations = [
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
        {"op": "filter", "column": "sum_x", "cmp": ">=", "value": 3},
        {"op": "sort", "columns": ["g"], "ascending": True},
    ]

    assert legal_filter_above_groupby_positions([table], operations) == [0]
    detail = apply_filter_above_groupby([table], operations, random.Random(1))

    assert detail == "ir_pull_filter_above_groupby:0:x"
    assert operations[:2] == [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 3},
        {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
    ]
    case = Case("case-ir-groupby-pull", 1, [table], Program("prog-ir-groupby-pull", 1, operations))
    assert validate_case_program(case) == []


def test_mutate_case_can_select_ir_pull_filter_above_groupby_operator():
    table = TableData(
        "t0",
        [ColumnSpec("g", "str"), ColumnSpec("x", "int")],
        [{"g": "a", "x": 1}, {"g": "a", "x": 2}, {"g": "b", "x": 5}],
    )
    base = Case(
        "case-ir-groupby-pull-base",
        10,
        [table],
        Program(
            "prog-ir-groupby-pull-base",
            10,
            [
                {"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
                {"op": "filter", "column": "sum_x", "cmp": ">=", "value": 3},
            ],
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_pull_filter_above_groupby": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )

    assert result.metadata["mutation"]["operator"] == "ir_pull_filter_above_groupby"
    assert result.metadata["mutation"]["detail"].startswith("ir_pull_filter_above_groupby:")
    assert result.case.program.operations != base.program.operations
    assert validate_case_program(result.case) == []


def test_ir_wrap_with_window_inserts_valid_core_window_operation():
    table = TableData(
        "t0",
        [ColumnSpec("g", "str"), ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"g": "a", "x": 1, "y": 10}, {"g": "b", "x": 2, "y": 20}],
    )
    operations = [{"op": "filter", "column": "x", "cmp": ">=", "value": 1}]

    assert legal_window_wrap_positions([table], operations)
    detail = apply_wrap_with_window([table], operations, random.Random(1))

    assert detail.startswith("ir_wrap_with_window:")
    assert any(op["op"] in {"running_sum", "row_number_filter"} for op in operations)
    case = Case("case-ir-window-wrap", 1, [table], Program("prog-ir-window-wrap", 1, operations))
    assert validate_case_program(case) == []


def test_ir_wrap_with_window_reuses_prefix_states(monkeypatch):
    table = TableData(
        "t0",
        [ColumnSpec("g", "str"), ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"g": "a", "x": 1, "y": 10}, {"g": "b", "x": 2, "y": 20}],
    )
    operations = [{"op": "filter", "column": "x", "cmp": ">=", "value": 1}]

    def fail_state_after_operations(*args, **kwargs):
        del args, kwargs
        raise AssertionError("window wrap should reuse precomputed prefix states")

    monkeypatch.setattr(rewrite_groupby_module, "state_after_operations", fail_state_after_operations, raising=False)

    assert rewrite_groupby_module.legal_window_wrap_positions([table], operations)
    detail = rewrite_groupby_module.apply_wrap_with_window([table], operations, random.Random(1))

    assert detail.startswith("ir_wrap_with_window:")


def test_mutate_case_can_select_ir_wrap_with_window_operator():
    table = TableData(
        "t0",
        [ColumnSpec("g", "str"), ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"g": "a", "x": 1, "y": 10}, {"g": "b", "x": 2, "y": 20}],
    )
    base = Case(
        "case-ir-window-wrap-base",
        10,
        [table],
        Program("prog-ir-window-wrap-base", 10, [{"op": "filter", "column": "x", "cmp": ">=", "value": 1}]),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_wrap_with_window": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )

    assert result.metadata["mutation"]["operator"] == "ir_wrap_with_window"
    assert result.metadata["mutation"]["detail"].startswith("ir_wrap_with_window:")
    assert result.case.program.operations != base.program.operations
    assert validate_case_program(result.case) == []


def test_ir_subtree_splice_moves_local_operation_subtree_to_legal_position():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 10}, {"x": 2, "y": 20}],
    )
    operations = [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "columns": ["y"], "ascending": True},
        {"op": "limit", "n": 2},
    ]

    positions = legal_subtree_splice_positions([table], operations)
    assert positions

    detail = apply_subtree_splice([table], operations, random.Random(3))

    assert detail.startswith("ir_splice_subtree:")
    assert operations != [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "columns": ["y"], "ascending": True},
        {"op": "limit", "n": 2},
    ]
    case = Case("case-ir-splice", 1, [table], Program("prog-ir-splice", 1, operations))
    assert validate_case_program(case) == []


def test_legal_subtree_splice_positions_reuses_prefix_states(monkeypatch):
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 10}, {"x": 2, "y": 20}],
    )
    operations = [
        {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
        {"op": "sort", "columns": ["y"], "ascending": True},
        {"op": "limit", "n": 2},
    ]

    def fail_state_after_operations(*args, **kwargs):
        del args, kwargs
        raise AssertionError("splice position enumeration should reuse precomputed prefix states")

    monkeypatch.setattr(rewrite_splice_module, "state_after_operations", fail_state_after_operations)

    positions = rewrite_splice_module.legal_subtree_splice_positions([table], operations)

    assert positions


def test_mutate_case_can_select_ir_subtree_splice_operator():
    table = TableData(
        "t0",
        [ColumnSpec("x", "int"), ColumnSpec("y", "int")],
        [{"x": 1, "y": 10}, {"x": 2, "y": 20}],
    )
    base = Case(
        "case-ir-splice-base",
        10,
        [table],
        Program(
            "prog-ir-splice-base",
            10,
            [
                {"op": "filter", "column": "x", "cmp": ">=", "value": 1},
                {"op": "sort", "columns": ["y"], "ascending": True},
                {"op": "limit", "n": 2},
            ],
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_splice_subtree": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )

    assert result.metadata["mutation"]["operator"] == "ir_splice_subtree"
    assert result.metadata["mutation"]["detail"].startswith("ir_splice_subtree:")
    assert result.case.program.operations != base.program.operations
    assert validate_case_program(result.case) == []


def test_ir_redundant_fold_merges_adjacent_idempotent_operations():
    table = TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("y", "int")], [{"x": 1, "y": 2}])
    operations = [
        {"op": "select", "columns": ["x", "y"]},
        {"op": "select", "columns": ["x"]},
        {"op": "limit", "n": 10},
        {"op": "limit", "n": 3},
    ]

    assert legal_redundant_fold_positions([table], operations) == [0, 2]
    select_detail = apply_redundant_op_fold([table], operations, random.Random(1))
    limit_detail = apply_redundant_op_fold([table], operations, random.Random(1))

    assert select_detail == "ir_fold_redundant_op:0:select+select"
    assert limit_detail == "ir_fold_redundant_op:1:limit+limit"
    assert operations == [
        {"op": "select", "columns": ["x"]},
        {"op": "limit", "n": 3},
    ]


def test_mutate_case_can_select_ir_redundant_fold_operator():
    table = TableData("t0", [ColumnSpec("x", "int"), ColumnSpec("y", "int")], [{"x": 1, "y": 2}])
    base = Case(
        "case-ir-fold-base",
        10,
        [table],
        Program(
            "prog-ir-fold-base",
            10,
            [
                {"op": "select", "columns": ["x", "y"]},
                {"op": "select", "columns": ["x"]},
                {"op": "limit", "n": 10},
            ],
        ),
    )

    result = mutate_case_with_metadata(
        base,
        99,
        operator_scores={"ir_fold_redundant_op": 5.0, "__untried__": -5.0},
        allow_probe_operators=False,
    )

    assert result.metadata["mutation"]["operator"] == "ir_fold_redundant_op"
    assert result.metadata["mutation"]["detail"].startswith("ir_fold_redundant_op:")
    assert result.case.program.operations != base.program.operations
    assert validate_case_program(result.case) == []


def test_fallback_column_type_treats_probe_outputs_as_boolean():
    assert _fallback_column_type("unexpected_else_seen") == "bool"
    assert _fallback_column_type("sorted_ok_x") == "bool"
    assert _fallback_column_type("setop_all_duplicate_mismatch") == "bool"
    assert _fallback_column_type("rolling_mean_by_null_count_mismatch_1") == "bool"


def test_shrink_mutators_fold_redundant_operations():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])
    operations = [
        {"op": "sort", "columns": ["x"], "ascending": True},
        {"op": "sort", "columns": ["x"], "ascending": False},
        {"op": "limit", "n": 10},
        {"op": "limit", "n": 3},
    ]

    sort_detail = shrink_fold_redundant_op([table], operations, random.Random(1))
    limit_detail = shrink_fold_redundant_op([table], operations, random.Random(1))

    assert sort_detail == "shrink_fold_redundant_op:sort:sort"
    assert limit_detail == "shrink_fold_redundant_op:limit"
    assert operations == [
        {"op": "sort", "columns": ["x"], "ascending": False},
        {"op": "limit", "n": 3},
    ]


def test_shrink_mutators_merge_filters_and_drop_tail():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])
    operations = [
        {"op": "filter", "column": "x", "cmp": "range_closed", "value": [0, 10]},
        {"op": "filter", "column": "x", "cmp": "range_closed", "value": [2, 8]},
        {"op": "select", "columns": ["x"]},
    ]

    merge_detail = shrink_merge_adjacent_filters([table], operations, random.Random(1))
    drop_detail = shrink_drop_tail_op([table], operations, random.Random(1))

    assert merge_detail == "shrink_merge_adjacent_filters:range_closed"
    assert operations[0] == {"op": "filter", "column": "x", "cmp": "range_closed", "value": [2, 8]}
    assert drop_detail == "shrink_drop_tail_op:select"
    assert len(operations) == 1


def test_shrink_mutator_inlines_single_use_mutate_alias():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])
    operations = [
        {"op": "mutate", "column": "m_x", "expr": {"kind": "add_const", "source": "x", "value": 1}},
        {"op": "filter", "column": "m_x", "cmp": ">=", "value": 0},
        {"op": "select", "columns": ["x"]},
    ]

    detail = shrink_inline_single_use_mutate([table], operations, random.Random(1))

    assert detail == "shrink_inline_single_use_mutate:m_x->x"
    assert operations[0] == {"op": "filter", "column": "x", "cmp": ">=", "value": 0}
    assert all(op.get("op") != "mutate" for op in operations)


def test_shrink_mutator_inlines_alias_inside_select_columns():
    table = TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": 2}])
    operations = [
        {"op": "mutate", "column": "m_x", "expr": {"kind": "add_const", "source": "x", "value": 1}},
        {"op": "select", "columns": ["m_x"]},
    ]

    detail = shrink_inline_single_use_mutate([table], operations, random.Random(1))

    assert detail == "shrink_inline_single_use_mutate:m_x->x"
    assert operations == [{"op": "select", "columns": ["x"]}]


def test_mutate_case_preserves_issue_profile_metadata_for_guidance():
    base = Case(
        "case-template",
        10,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program("prog-template", 10, [{"op": "limit", "n": 2}]),
        metadata={
            "generator_profile": "discovery",
            "mixed_generator_profile": "post_topk_range_filter",
            "source_issue": "https://github.com/example/project/issues/1",
        },
    )

    result = mutate_case_with_metadata(base, 99)

    assert result.metadata["candidate_source"] == "feedback_mutation"
    assert result.metadata["generator_profile"] == "discovery"
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
    assert {"value", "nullify_value", "duplicate_row", "drop_row", "shuffle_rows"}.issubset(
        TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    )
    assert {"value", "nullify_value", "duplicate_row", "drop_row", "shuffle_rows"} == set(
        TABLE_ONLY_MUTATION_OPERATOR_NAMES
    )
    assert {"append_op", "drop_op", "tweak_op"}.issubset(MUTATION_OPERATOR_NAMES)
    assert {
        "ir_swap_adjacent",
        "ir_pushdown_filter",
        "ir_pull_filter_above_groupby",
        "ir_wrap_with_window",
        "ir_splice_subtree",
        "ir_fold_redundant_op",
        "shrink_drop_tail_op",
        "shrink_fold_redundant_op",
        "shrink_merge_adjacent_filters",
        "shrink_inline_single_use_mutate",
    }.issubset(MUTATION_OPERATOR_NAMES)
    assert "ir_swap_adjacent" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "ir_pushdown_filter" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "ir_pull_filter_above_groupby" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "ir_wrap_with_window" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "ir_splice_subtree" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "ir_fold_redundant_op" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "shrink_fold_redundant_op" in DISCOVERY_MUTATION_OPERATOR_NAMES
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
    assert "append_float_literal_precision_probe" in MUTATION_OPERATOR_NAMES
    assert "append_timestamp_precision_filter_probe" in MUTATION_OPERATOR_NAMES


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
    assert "append_arrow_bool_groupby_reduction_probe" in MUTATION_OPERATOR_NAMES
    assert "append_dataset_isin_all_match_probe" in MUTATION_OPERATOR_NAMES
    assert "append_large_string_partition_probe" in MUTATION_OPERATOR_NAMES
    assert "append_hash_pivot_wider_probe" in MUTATION_OPERATOR_NAMES
    assert "append_list_flatten_parent_indices_probe" in MUTATION_OPERATOR_NAMES
    assert "append_rolling_mean_by_null_count_probe" in MUTATION_OPERATOR_NAMES
    assert "append_csv_long_numeric_roundtrip_probe" in MUTATION_OPERATOR_NAMES
    assert "append_grouped_topk" in MUTATION_OPERATOR_NAMES
    assert "append_groupby_fractional_membership_filter" in MUTATION_OPERATOR_NAMES
    assert "append_groupby_fractional_membership_filter" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_normalized_string_membership" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_sql_distinct_null_topk" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_coalesce_membership" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_coalesce_membership" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_left_join_coalesce_membership" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_case_membership" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_case_membership" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_left_join_case_membership" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_empty_filter_global_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_sql_union_coalesce_distinct_topk" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_sql_union_coalesce_distinct_topk" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_sql_union_coalesce_distinct_topk" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_coalesce_sort_topk" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_boolean_membership_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_boolean_membership_case_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_boolean_membership_case_aggregate" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_case_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_case_aggregate" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_case_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_case_aggregate" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_boolean_antijoin_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_boolean_antijoin_case_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_boolean_antijoin_case_aggregate" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_filter_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_filter_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_left_join_boolean_coalesce_filter_aggregate" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_numeric_text_boolean_antijoin_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_numeric_text_boolean_antijoin_case_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_numeric_text_boolean_antijoin_case_aggregate" not in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_multi_key_membership_case_aggregate" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_multi_key_membership_case_aggregate" in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_multi_key_membership_case_aggregate" in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES
    assert "append_join_filter_groupby_topk" in DISCOVERY_MUTATION_OPERATOR_NAMES
    assert "append_join_filter_groupby_topk" not in TABLE_MUTATING_MUTATION_OPERATOR_NAMES
    assert "append_join_filter_groupby_topk" not in TABLE_APPEND_ONLY_MUTATION_OPERATOR_NAMES


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


def test_append_float_literal_precision_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("x", "float")],
        [{"id": 0, "x": 0.1}],
    )
    operations = [{"op": "select", "columns": ["id"]}]

    detail = _append_float_literal_precision_probe([table], operations, random.Random(1))

    assert detail.startswith("append_float_literal_precision_probe:")
    assert operations[-1]["op"] == "float_literal_precision_probe"
    assert operations[-1]["as"] == "float_literal_precision_mismatch"
    assert operations[-1]["literal"] in {
        "0.10000000000000001",
        "0.29999999999999999",
        "1.2345678901234567",
        "9007199254740993.0",
    }
    case = Case(
        "case-mut-float-literal-precision",
        1,
        [table],
        Program("prog-mut-float-literal-precision", 1, operations),
    )
    assert validate_case_program(case) == []


def test_append_timestamp_precision_filter_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("ts", "str")],
        [{"id": 0, "ts": "2024-01-01T00:00:00.000000001"}],
    )
    operations = [{"op": "select", "columns": ["ts"]}]

    detail = _append_timestamp_precision_filter_probe([table], operations, random.Random(1))

    assert detail.startswith("append_timestamp_precision_filter_probe:")
    assert operations[-1] == {
        "op": "timestamp_precision_filter_probe",
        "as": "timestamp_precision_filter_mismatch",
    }
    case = Case(
        "case-mut-timestamp-precision-filter",
        1,
        [table],
        Program("prog-mut-timestamp-precision-filter", 1, operations),
    )
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


def test_append_arrow_bool_groupby_reduction_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("flag", "bool")],
        [{"id": 0, "flag": True}, {"id": 1, "flag": None}],
    )
    operations = [{"op": "select", "columns": ["flag"]}]

    detail = _append_arrow_bool_groupby_reduction_probe([table], operations, random.Random(1))

    assert detail.startswith("append_arrow_bool_groupby_reduction_probe:")
    assert operations[-1] == {
        "op": "arrow_bool_groupby_reduction_probe",
        "as": "arrow_bool_groupby_reduction_mismatch",
    }
    case = Case(
        "case-mut-arrow-bool-groupby-reduction",
        1,
        [table],
        Program("prog-mut-arrow-bool-groupby-reduction", 1, operations),
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


def test_append_csv_long_numeric_roundtrip_probe_mutation_stays_valid():
    table = TableData(
        "t0",
        [ColumnSpec("id", "int"), ColumnSpec("raw", "str")],
        [{"id": 0, "raw": "9007199254740993"}],
    )
    operations = [{"op": "select", "columns": ["raw"]}]

    detail = _append_csv_long_numeric_roundtrip_probe([table], operations, random.Random(1))

    assert detail.startswith("append_csv_long_numeric_roundtrip_probe:")
    assert operations[-1]["op"] == "csv_long_numeric_roundtrip_probe"
    assert operations[-1]["as"] == "csv_long_numeric_roundtrip_mismatch"
    assert 3 <= len(operations[-1]["values"]) <= 6
    assert all(isinstance(value, str) and value.isdigit() for value in operations[-1]["values"])
    case = Case(
        "case-mut-csv-long-numeric-roundtrip",
        1,
        [table],
        Program("prog-mut-csv-long-numeric-roundtrip", 1, operations),
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


def test_append_coalesce_sort_topk_mutation_stays_valid_and_discoverable():
    table = TableData(
        "t0",
        [
            ColumnSpec("id", "int"),
            ColumnSpec("x", "int", nullable=True),
            ColumnSpec("y", "int", nullable=True),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("s", "str", nullable=True),
        ],
        [
            {"id": 0, "x": 2, "y": None, "g": "b", "s": None},
            {"id": 1, "x": None, "y": 1, "g": None, "s": "a"},
            {"id": 2, "x": 3, "y": 3, "g": "c", "s": "c"},
        ],
    )
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_coalesce_sort_topk([table], operations, random.Random(1))

    assert detail.startswith("append_coalesce_sort_topk:")
    assert [op["op"] for op in operations[-3:]] == ["coalesce", "sort", "limit"]
    assert operations[-2]["keys"][0]["column"] == operations[-3]["as"]
    case = Case("case-mut-coalesce-topk", 1, [table], Program("prog-mut-coalesce-topk", 1, operations))
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


def test_append_join_filter_groupby_topk_mutation_stays_valid():
    left = TableData(
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
            {"id": 2, "g": "a", "x": 3, "flag": False},
            {"id": 3, "g": None, "x": 5, "flag": True},
        ],
    )
    right = TableData(
        "t1",
        [ColumnSpec("id", "int"), ColumnSpec("g", "str")],
        [{"id": 0, "g": "a"}, {"id": 2, "g": "a"}, {"id": 3, "g": None}],
    )
    tables = [left, right]
    operations = [{"op": "filter", "column": "id", "cmp": ">=", "value": 0}]

    detail = _append_join_filter_groupby_topk(tables, operations, random.Random(1))

    assert detail.startswith("append_join_filter_groupby_topk:")
    assert [op["op"] for op in operations[-6:]] == ["join", "filter", "groupby", "select", "sort", "limit"]
    assert isinstance(operations[-6]["left_on"], list)
    assert operations[-4]["keys"] == operations[-3]["columns"][: len(operations[-4]["keys"])]
    assert operations[-2]["keys"][0]["column"] == operations[-4]["aggs"][0]["as"]
    assert len(tables) == 2
    case = Case(
        "case-mut-join-filter-groupby-topk",
        1,
        tables,
        Program("prog-mut-join-filter-groupby-topk", 1, operations),
    )
    assert validate_case_program(case) == []
