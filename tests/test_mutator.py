import random

from datadiff.classification_oracle import validate_case_program
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.mutator import (
    MUTATION_OPERATOR_NAMES,
    _append_group_quantile_probe,
    _append_scalar_subquery_probe,
    _append_window_avg_probe,
    _append_struct_distinct_probe,
    _append_boolean_predicate_filter_probe,
    _append_grouped_topk_probe,
    _append_order_projection_probe,
    _append_range_filter_probe,
    _append_random_case_probe,
    _append_running_sum_probe,
    _append_sortedness_check_probe,
    _append_truth_filter_probe,
    _append_tuple_absence_filter_probe,
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

    for seed in range(200):
        op = _random_operation([table], [], random.Random(seed))
        if op is None or op["op"] != "mutate":
            continue
        assert op["expr"]["kind"] in {"string_length", "string_lower"}
        assert op["expr"]["source"] == "s"


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


def test_mutation_operator_registry_covers_row_value_and_operation_mutations():
    assert {"value", "nullify_value", "duplicate_row", "drop_row", "shuffle_rows"}.issubset(MUTATION_OPERATOR_NAMES)
    assert {"append_op", "drop_op", "tweak_op"}.issubset(MUTATION_OPERATOR_NAMES)
    assert "append_order_projection" in MUTATION_OPERATOR_NAMES
    assert "append_truth_filter" in MUTATION_OPERATOR_NAMES
    assert "append_boolean_predicate_filter" in MUTATION_OPERATOR_NAMES
    assert "append_range_filter" in MUTATION_OPERATOR_NAMES
    assert "append_tuple_absence_filter" in MUTATION_OPERATOR_NAMES
    assert "append_running_sum" in MUTATION_OPERATOR_NAMES
    assert "append_sortedness_check" in MUTATION_OPERATOR_NAMES
    assert "append_random_case_probe" in MUTATION_OPERATOR_NAMES
    assert "append_group_quantile_probe" in MUTATION_OPERATOR_NAMES
    assert "append_scalar_subquery_probe" in MUTATION_OPERATOR_NAMES
    assert "append_window_avg_probe" in MUTATION_OPERATOR_NAMES
    assert "append_struct_distinct_probe" in MUTATION_OPERATOR_NAMES
    assert "append_grouped_topk" in MUTATION_OPERATOR_NAMES


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
