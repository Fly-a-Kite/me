"""Smoke tests for the chDB backend adapter.

chDB is an optional dependency; if not installed, this test file's body
short-circuits so the suite remains green in lean environments. Tests that
require live SQL execution are guarded by `skip_if_chdb_missing`.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


def _chdb_available() -> bool:
    return importlib.util.find_spec("chdb") is not None


skip_if_chdb_missing = pytest.mark.skipif(
    not _chdb_available(),
    reason="chdb package not installed; run `pip install chdb` to enable.",
)


def test_chdb_backend_module_imports() -> None:
    """Adapter module must import even without chdb installed."""
    module = importlib.import_module("datadiff.backends.chdb_backend")
    assert hasattr(module, "ChDBBackend")
    assert hasattr(module, "CHDB_DIALECT")


def test_chdb_target_registered() -> None:
    from datadiff.targets import TARGETS, target_spec

    assert "chdb" in TARGETS
    spec = target_spec("chdb")
    assert spec.adapter == "datadiff.backends.chdb_backend.ChDBBackend"
    assert spec.status == "experimental"
    assert spec.family == "embedded_olap"
    assert "op:filter" in spec.capabilities
    assert "op:running_sum" in spec.capabilities
    assert "running:partition_by" in spec.capabilities
    # Unsupported stateful ops must remain absent.
    assert "op:row_number_filter" not in spec.capabilities


def test_chdb_suites_contain_chdb() -> None:
    from datadiff.targets import TARGET_SUITES

    for suite in ("chdb_cross", "chdb_olap_cross", "latest_with_chdb"):
        assert suite in TARGET_SUITES, suite
        assert "chdb" in TARGET_SUITES[suite], suite


def test_chdb_dialect_uses_clickhouse_types() -> None:
    from datadiff.backends.chdb_backend import CHDB_DIALECT

    assert CHDB_DIALECT.cast_type("bool") == "Nullable(Bool)"
    assert CHDB_DIALECT.cast_type("float") == "Nullable(Float64)"
    assert CHDB_DIALECT.cast_type("int") == "Nullable(Int64)"
    assert CHDB_DIALECT.cast_type("str") == "Nullable(String)"
    # division must cast through Float64
    assert "toFloat64" in CHDB_DIALECT.division_sql("a", "b")


def test_chdb_quote_uses_backticks() -> None:
    from datadiff.backends.chdb_backend import _quote

    assert _quote("col_name") == "`col_name`"
    # escapes embedded backticks
    assert _quote("a`b") == "`a``b`"


def test_chdb_lit_preserves_float64_bits() -> None:
    from datadiff.backends.chdb_backend import _lit

    assert _lit(None) == "NULL"
    assert _lit(True) == "true"
    assert _lit(False) == "false"
    assert _lit(42) == "42"
    assert _lit(15.694) == "reinterpretAsFloat64(unhex('17d9cef753632f40'))"
    assert _lit(-0.0) == "reinterpretAsFloat64(unhex('0000000000000080'))"
    assert _lit(float("inf")) == "reinterpretAsFloat64(unhex('000000000000f07f'))"
    assert _lit(float("-inf")) == "reinterpretAsFloat64(unhex('000000000000f0ff'))"
    assert _lit(float("nan")) == "reinterpretAsFloat64(unhex('000000000000f87f'))"
    assert _lit("o'reilly") == "'o\\'reilly'"


def test_chdb_ch_type_wraps_in_nullable() -> None:
    from datadiff.backends.chdb_backend import _ch_type

    assert _ch_type("int") == "Nullable(Int64)"
    assert _ch_type("float") == "Nullable(Float64)"
    assert _ch_type("bool") == "Nullable(Bool)"
    assert _ch_type("str") == "Nullable(String)"


def test_chdb_backend_returns_error_when_chdb_missing(monkeypatch) -> None:
    """If chdb is unavailable, run() returns BackendResult(status='error')
    rather than crashing the whole batch."""
    if _chdb_available():
        pytest.skip("chdb is installed; this guard test only runs without it.")
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    backend = ChDBBackend()
    table = TableData(
        name="t0",
        columns=[ColumnSpec(name="a", type="int")],
        rows=[{"a": 1}],
    )
    program = Program(program_id="p", seed=0, operations=[])
    result = backend.run([table], program)
    # Either the chdb session open fails, or the pandas import fails — both
    # are caught and surfaced as a clean BackendResult(status='error') rather
    # than an uncaught exception.
    assert result.status == "error"
    assert result.error_type


@skip_if_chdb_missing
def test_chdb_basic_select_smoke() -> None:
    """End-to-end smoke: create one table, run identity SELECT, expect 1 row.

    Skipped unless `chdb` is installed.
    """
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    backend = ChDBBackend()
    table = TableData(
        name="t0",
        columns=[
            ColumnSpec(name="a", type="int"),
            ColumnSpec(name="b", type="str"),
        ],
        rows=[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
    )
    program = Program(program_id="p_smoke", seed=0, operations=[])
    result = backend.run([table], program)
    assert result.status == "ok", f"chdb run failed: {result.error_type}: {result.error}"
    assert result.data is not None
    assert len(result.data) == 2


@skip_if_chdb_missing
def test_chdb_filter_and_sort_smoke() -> None:
    """Smoke: filter + sort returns deterministic ordered rows."""
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    backend = ChDBBackend()
    table = TableData(
        name="t0",
        columns=[ColumnSpec(name="a", type="int")],
        rows=[{"a": 3}, {"a": 1}, {"a": 2}],
    )
    program = Program(
        program_id="p_filter_sort",
        seed=0,
        operations=[
            {"kind": "filter", "column": "a", "cmp": ">=", "value": 2},
            {
                "kind": "sort",
                "keys": [{"column": "a", "ascending": True, "nulls": "last"}],
            },
        ],
    )
    result = backend.run([table], program)
    assert result.status == "ok", f"chdb run failed: {result.error_type}: {result.error}"
    values = [int(row) for row in result.data["a"].tolist()]
    assert values == [2, 3]


@skip_if_chdb_missing
def test_chdb_nullable_cast_preserves_nulls() -> None:
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    table = TableData(
        "t0",
        [ColumnSpec("x", "int")],
        [{"x": None}, {"x": 7}],
    )
    program = Program(
        "p_nullable_cast",
        0,
        [
            {
                "op": "mutate",
                "column": "label",
                "expr": {"kind": "cast", "source": "x", "to": "str"},
            },
            {"op": "select", "columns": ["label"]},
        ],
    )

    result = ChDBBackend().run([table], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert result.data["label"].tolist() == [None, "7"]


@skip_if_chdb_missing
def test_chdb_join_avoids_duplicate_right_column_names() -> None:
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    left = TableData(
        "t0",
        [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
        [{"id": 1, "tag": "a"}],
    )
    right = TableData(
        "t1",
        [ColumnSpec("id", "int", nullable=False), ColumnSpec("tag", "str")],
        [{"id": 99, "tag": "a"}],
    )
    program = Program(
        "p_duplicate_join_column",
        0,
        [{"op": "join", "table": "t1", "left_on": "tag", "right_on": "tag", "how": "left"}],
    )

    result = ChDBBackend().run([left, right], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert result.data["id"].tolist() == [1]


@skip_if_chdb_missing
def test_chdb_count_output_can_join_int64_key() -> None:
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    left = TableData(
        "t0",
        [ColumnSpec("g", "str"), ColumnSpec("id", "int", nullable=False)],
        [{"g": "a", "id": 1}, {"g": "a", "id": 2}],
    )
    right = TableData(
        "t1",
        [ColumnSpec("id", "int", nullable=False), ColumnSpec("label", "str")],
        [{"id": 2, "label": "two"}],
    )
    program = Program(
        "p_count_join",
        0,
        [
            {
                "op": "groupby",
                "keys": ["g"],
                "aggs": [{"column": "id", "func": "count", "as": "count_id"}],
            },
            {"op": "join", "table": "t1", "left_on": "count_id", "right_on": "id", "how": "left"},
        ],
    )

    result = ChDBBackend().run([left, right], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert result.data["label"].tolist() == ["two"]


@skip_if_chdb_missing
def test_chdb_partitioned_running_sum_preserves_null_semantics_and_order() -> None:
    from datadiff.backends.chdb_backend import ChDBBackend
    from datadiff.dsl import ColumnSpec, Program, TableData

    table = TableData(
        "t0",
        [
            ColumnSpec("grp", "str"),
            ColumnSpec("seq", "int", nullable=False),
            ColumnSpec("x", "float"),
        ],
        [
            {"grp": "b", "seq": 1, "x": 2.0},
            {"grp": "a", "seq": 0, "x": None},
            {"grp": "b", "seq": 0, "x": None},
            {"grp": "a", "seq": 1, "x": 1.5},
            {"grp": "a", "seq": 2, "x": None},
        ],
    )
    program = Program(
        "p-chdb-partitioned-running-sum",
        0,
        [
            {
                "op": "running_sum",
                "source": "x",
                "column": "run_x",
                "partition_by": ["grp"],
                "order_by": [
                    {"column": "seq", "ascending": True, "nulls": "last"}
                ],
            }
        ],
    )

    result = ChDBBackend().run([table], program)

    assert result.status == "ok", f"{result.error_type}: {result.error}"
    assert result.data["grp"].tolist() == ["a", "a", "a", "b", "b"]
    assert result.data["seq"].tolist() == [0, 1, 2, 0, 1]
    assert result.data["run_x"].tolist() == [None, 1.5, 1.5, None, 2.0]
