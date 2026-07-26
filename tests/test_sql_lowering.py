from __future__ import annotations

from random import Random

import pytest

from datadiff.backends.chdb_backend import CHDB_DIALECT
from datadiff.backends.duckdb_backend import DUCKDB_DIALECT
from datadiff.backends.sql_lowering import (
    HarnessLoweringError,
    SqlPipelineState,
    render_filter_condition,
    render_join_condition,
    render_join_right_projection,
    render_mutate_expr,
)
from datadiff.dsl import SortKey


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def test_sql_pipeline_state_select_preserves_pending_order_with_hidden_projection():
    state = SqlPipelineState.from_columns(["a", "b", "c"])
    state.pending_order = [SortKey("b", True, "last")]

    projection = state.select_with_pending_order(["a"], _quote)

    assert projection == 'q."a", q."b" AS "__datadiff_order_0_0"'
    assert state.current_cols == ["a", "__datadiff_order_0_0"]
    assert state.visible_cols == ["a"]
    assert state.hidden_order_cols == ["__datadiff_order_0_0"]
    assert state.pending_order == [SortKey("__datadiff_order_0_0", True, "last")]


def test_sql_pipeline_state_freeze_and_drop_hidden_order_cols_are_stable():
    state = SqlPipelineState.from_columns(["a", "b"])
    state.pending_order = [SortKey("b", False, "first")]
    state.select_with_pending_order(["a"], _quote)

    frozen = state.freeze_pending_order()

    assert frozen == (
        "__datadiff_order_freeze_1",
        [SortKey("__datadiff_order_0_0", False, "first")],
    )
    assert state.current_cols == ["a", "__datadiff_order_0_0", "__datadiff_order_freeze_1"]
    assert state.hidden_order_cols == ["__datadiff_order_0_0", "__datadiff_order_freeze_1"]
    assert state.pending_order == [SortKey("__datadiff_order_freeze_1", True, "last")]

    dropped = state.drop_hidden_order_cols()

    assert dropped is True
    assert state.current_cols == ["a"]
    assert state.visible_cols == ["a"]
    assert state.hidden_order_cols == []
    assert state.pending_order is None


def test_sql_pipeline_state_rejects_stale_pending_order_aliases():
    state = SqlPipelineState.from_columns(["a"])
    state.pending_order = [SortKey("missing", True, "last")]

    with pytest.raises(HarnessLoweringError, match="pending order references missing"):
        state.assert_invariants()


def test_sql_pipeline_state_reset_and_replace_projection_follow_visible_semantics():
    state = SqlPipelineState.from_columns(["x", "y"])
    projection = state.replace_projection_expr("x", 'ABS(q."x")', _quote)
    state.replace_visible_column("x")

    assert projection == 'q."y", ABS(q."x") AS "x"'
    assert state.current_cols == ["y", "x"]
    assert state.visible_cols == ["y", "x"]

    state.reset_to_alias("flag")

    assert state.current_cols == ["flag"]
    assert state.visible_cols == ["flag"]
    assert state.hidden_order_cols == []
    assert state.pending_order is None


def test_sql_pipeline_state_randomized_transitions_preserve_symbol_table_invariants():
    for seed in range(128):
        rnd = Random(seed)
        state = SqlPipelineState.from_columns(["a", "b", "c"])
        for _ in range(24):
            action = rnd.choice(("order", "select", "replace", "freeze", "drop", "clear"))
            if action == "order":
                state.pending_order = [
                    SortKey(rnd.choice(state.visible_cols), bool(rnd.randrange(2)), "last")
                ]
            elif action == "select":
                count = rnd.randint(1, len(state.visible_cols))
                state.select_with_pending_order(
                    rnd.sample(state.visible_cols, count),
                    _quote,
                )
            elif action == "replace":
                column = rnd.choice(state.visible_cols)
                state.replace_projection_expr(column, f'COALESCE(q.{_quote(column)}, 0)', _quote)
                state.replace_visible_column(column)
            elif action == "freeze":
                state.freeze_pending_order()
            elif action == "drop":
                state.drop_hidden_order_cols()
            else:
                state.clear_pending_order()
            state.assert_invariants()


def test_join_projection_uses_common_schema_contract_for_every_sql_dialect():
    projection, projected = render_join_right_projection(
        ["id", "join_key", "value", "tag"],
        ["join_key"],
        ["id", "left_value"],
        _quote,
    )

    assert projected == ["value", "tag"]
    assert projection == ', r."value" AS "value", r."tag" AS "tag"'


def test_join_condition_aligns_both_keys_to_declared_logical_type():
    condition = render_join_condition(
        {"op": "join", "left_on": "id", "right_on": "join_id"},
        CHDB_DIALECT,
        _quote,
        {"id": "int"},
        {"join_id": "int"},
    )

    assert condition == (
        'CAST(q."id" AS Nullable(Int64)) = '
        'CAST(r."join_id" AS Nullable(Int64))'
    )


def test_join_condition_fails_closed_on_logical_type_mismatch():
    with pytest.raises(ValueError, match="join key type mismatch"):
        render_join_condition(
            {"op": "join", "left_on": "id", "right_on": "join_id"},
            DUCKDB_DIALECT,
            _quote,
            {"id": "int"},
            {"join_id": "str"},
        )


def test_string_predicate_is_selected_by_dialect_not_backend_name():
    assert render_filter_condition('q."text"', "'A'", "str_contains", CHDB_DIALECT) == (
        'CASE WHEN q."text" IS NULL THEN NULL ELSE position(q."text", \'A\') > 0 END'
    )
    assert render_filter_condition('q."text"', "'A'", "str_contains", DUCKDB_DIALECT) == (
        'CASE WHEN q."text" IS NULL THEN NULL ELSE INSTR(q."text", \'A\') > 0 END'
    )


def test_string_length_uses_declared_character_semantics_per_dialect():
    operation = {
        "op": "mutate",
        "column": "length",
        "expr": {"kind": "string_length", "source": "text"},
    }

    assert render_mutate_expr(operation, CHDB_DIALECT, _quote, repr)[1] == 'lengthUTF8(q."text")'
    assert render_mutate_expr(operation, DUCKDB_DIALECT, _quote, repr)[1] == 'LENGTH(q."text")'
