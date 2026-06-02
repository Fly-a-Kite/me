from __future__ import annotations

from datadiff.backends.sql_lowering import SqlPipelineState
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
    assert state.pending_order == [SortKey("__datadiff_order_freeze_1", True, "last")]


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
