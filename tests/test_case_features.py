from datadiff.case_features import (
    case_contains_inf,
    case_contains_nan,
    case_contains_non_ascii_string,
    case_contains_null,
    case_contains_special_float,
    case_has_null_filter_literal,
    case_has_outer_join_truth_filter,
    case_has_path_projection_keyed_pick,
    case_has_post_topk_filter,
    case_has_running_sum,
    case_uses_modulo,
    case_uses_unicode_case_mapping,
    last_probe_root,
)
from datadiff.dsl import Case, ColumnSpec, Program, TableData


def _case(seed: int, operations: list[dict], rows: list[dict] | None = None) -> Case:
    return Case(
        case_id=f"case-{seed}",
        seed=seed,
        tables=[
            TableData(
                "t0",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("s", "str"),
                    ColumnSpec("x", "float"),
                    ColumnSpec("flag", "bool"),
                ],
                rows
                or [
                    {"id": 0, "s": "alpha", "x": 1.5, "flag": True},
                    {"id": 1, "s": "中文", "x": None, "flag": None},
                    {"id": 2, "s": "", "x": float("nan"), "flag": False},
                ],
            ),
            TableData(
                "t1",
                [
                    ColumnSpec("id", "int", nullable=False),
                    ColumnSpec("flag_r", "bool"),
                ],
                [
                    {"id": 0, "flag_r": True},
                    {"id": 1, "flag_r": None},
                ],
            ),
        ],
        program=Program(program_id=f"prog-{seed}", seed=seed, operations=operations),
    )


def test_case_features_track_data_boundaries():
    case = _case(1, [])

    assert case_contains_null(case) is True
    assert case_contains_nan(case) is True
    assert case_contains_inf(case) is False
    assert case_contains_special_float(case) is True
    assert case_contains_non_ascii_string(case) is True


def test_case_features_track_unicode_modulo_running_sum_and_path_pick():
    case = _case(
        2,
        [
            {"op": "mutate", "column": "s_upper", "expr": {"kind": "string_upper", "source": "s"}},
            {"op": "mutate", "column": "x_mod", "expr": {"kind": "arith_const", "source": "x", "op": "mod", "value": 2}},
            {"op": "running_sum", "column": "x", "as": "running_x"},
            {"op": "mutate", "column": "base", "expr": {"kind": "string_basename", "source": "s"}},
            {"op": "row_number_filter", "partition_by": ["base"], "order_by": [{"column": "id"}], "n": 1},
        ],
    )

    assert case_uses_unicode_case_mapping(case) is True
    assert case_uses_modulo(case) is True
    assert case_has_running_sum(case) is True
    assert case_has_path_projection_keyed_pick(case) is True


def test_last_probe_root_uses_latest_probe_operation():
    case = _case(
        3,
        [
            {"op": "random_case_probe"},
            {"op": "window_avg_probe"},
            {"op": "round_even_probe"},
        ],
    )

    assert last_probe_root(case) == "round_even_float_scale"


def test_case_has_null_filter_literal_ignores_explicit_null_predicates():
    noisy = _case(4, [{"op": "filter", "column": "s", "cmp": "==", "value": None}])
    explicit_null = _case(5, [{"op": "filter", "column": "s", "cmp": "is_null", "value": None}])
    truth_predicate = _case(6, [{"op": "filter", "column": "flag", "cmp": "bool_is_true", "value": None}])

    assert case_has_null_filter_literal(noisy) is True
    assert case_has_null_filter_literal(explicit_null) is False
    assert case_has_null_filter_literal(truth_predicate) is False


def test_case_has_outer_join_truth_filter_requires_left_join_boundary():
    positive = _case(
        7,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "filter", "column": "flag_r", "cmp": "bool_is_not_true", "value": None},
        ],
    )
    reset = _case(
        8,
        [
            {"op": "join", "table": "t1", "left_on": "id", "right_on": "id", "how": "left"},
            {"op": "groupby", "keys": ["id"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]},
            {"op": "filter", "column": "sum_x", "cmp": "bool_is_not_true", "value": None},
        ],
    )

    assert case_has_outer_join_truth_filter(positive) is True
    assert case_has_outer_join_truth_filter(reset) is False


def test_case_has_post_topk_filter_tracks_filter_after_sort_and_topk():
    positive = _case(9, [{"op": "sort", "columns": ["x"]}, {"op": "limit", "n": 2}, {"op": "filter", "column": "x", "cmp": ">", "value": 0.0}])
    negative = _case(10, [{"op": "sort", "columns": ["x"]}, {"op": "filter", "column": "x", "cmp": ">", "value": 0.0}, {"op": "limit", "n": 2}])

    assert case_has_post_topk_filter(positive) is True
    assert case_has_post_topk_filter(negative) is False
