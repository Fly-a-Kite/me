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
    # v1 disabled ops must be absent
    assert "op:running_sum" not in spec.capabilities
    assert "op:row_number_filter" not in spec.capabilities


def test_chdb_suites_contain_chdb() -> None:
    from datadiff.targets import TARGET_SUITES

    for suite in ("chdb_cross", "chdb_olap_cross", "latest_with_chdb"):
        assert suite in TARGET_SUITES, suite
        assert "chdb" in TARGET_SUITES[suite], suite


def test_chdb_dialect_uses_clickhouse_types() -> None:
    from datadiff.backends.chdb_backend import CHDB_DIALECT

    assert CHDB_DIALECT.float_cast_type == "Float64"
    assert CHDB_DIALECT.int_cast_type == "Int64"
    assert CHDB_DIALECT.str_cast_type == "String"
    # division must cast through Float64
    assert "toFloat64" in CHDB_DIALECT.division_sql("a", "b")


def test_chdb_quote_uses_backticks() -> None:
    from datadiff.backends.chdb_backend import _quote

    assert _quote("col_name") == "`col_name`"
    # escapes embedded backticks
    assert _quote("a`b") == "`a``b`"


def test_chdb_lit_handles_special_floats() -> None:
    from datadiff.backends.chdb_backend import _lit

    assert _lit(None) == "NULL"
    assert _lit(True) == "true"
    assert _lit(False) == "false"
    assert _lit(42) == "42"
    assert _lit(float("inf")) == "inf"
    assert _lit(float("-inf")) == "-inf"
    assert _lit(float("nan")) == "nan"
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
