from __future__ import annotations

from datadiff.backends.base import Backend
from datadiff.backends.datafusion_backend import DataFusionBackend
from datadiff.backends.duckdb_backend import DuckDBBackend, DuckDBPersistentBackend
from datadiff.backends.faulty_backend import FaultyPandasBackend
from datadiff.backends.pandas_backend import PandasBackend
from datadiff.backends.polars_backend import PolarsBackend, PolarsLazyBackend, PolarsStreamingBackend
from datadiff.backends.pyarrow_backend import PyArrowBackend
from datadiff.backends.sqlite_backend import SQLiteBackend


def make_backend(name: str) -> Backend:
    if name == "pandas":
        return PandasBackend()
    if name == "polars":
        return PolarsBackend()
    if name == "polars_lazy":
        return PolarsLazyBackend()
    if name == "polars_streaming":
        return PolarsStreamingBackend()
    if name == "pyarrow":
        return PyArrowBackend()
    if name == "duckdb":
        return DuckDBBackend()
    if name == "duckdb_persistent":
        return DuckDBPersistentBackend()
    if name == "datafusion":
        return DataFusionBackend()
    if name == "sqlite":
        return SQLiteBackend()
    if name == "buggy_filter":
        return FaultyPandasBackend("buggy_filter", "filter")
    if name == "buggy_groupby":
        return FaultyPandasBackend("buggy_groupby", "groupby")
    if name == "buggy_join":
        return FaultyPandasBackend("buggy_join", "join")
    if name == "buggy_mutate":
        return FaultyPandasBackend("buggy_mutate", "mutate")
    raise ValueError(f"unknown backend: {name}")
