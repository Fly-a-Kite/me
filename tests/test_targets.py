import pytest

from datadiff.env import collect_environment
from datadiff.sqlite_runtime import SQLITE_RUNTIME, SQLITE_VERSION
from datadiff.targets import (
    common_capabilities,
    describe_targets,
    list_target_suites,
    resolve_target_backends,
    target_capability_matrix,
)


def test_resolve_target_suite_to_backends():
    assert resolve_target_backends(target_suite="dataframe") == ["pandas", "polars"]
    assert resolve_target_backends(target_suite="dataframe_lazy") == ["polars", "polars_lazy"]
    assert resolve_target_backends(target_suite="embedded_sql") == ["duckdb", "sqlite"]
    assert resolve_target_backends(target_suite="duckdb_storage_cross") == ["pandas", "duckdb_persistent"]
    assert resolve_target_backends(target_suite="cross_family") == ["pandas", "duckdb"]
    assert resolve_target_backends(target_suite="lazy_cross_family") == ["pandas", "polars_lazy", "duckdb"]
    assert resolve_target_backends(target_suite="core_lazy") == ["pandas", "polars", "polars_lazy", "duckdb", "sqlite"]
    assert resolve_target_backends(target_suite="datafusion_cross") == ["pandas", "duckdb", "datafusion"]
    assert resolve_target_backends(target_suite="arrow_cross") == ["pandas", "duckdb", "pyarrow"]
    assert resolve_target_backends(target_suite="latest_all_engines") == [
        "pandas",
        "pyarrow",
        "polars",
        "polars_lazy",
        "duckdb",
        "sqlite",
        "datafusion",
    ]
    assert resolve_target_backends(target_suite="seeded_filter") == ["pandas", "buggy_filter"]


def test_explicit_backends_override_suite_and_dedupe():
    assert resolve_target_backends("sqlite,pandas,sqlite", target_suite="dataframe") == ["sqlite", "pandas"]


def test_unknown_target_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown target backend"):
        resolve_target_backends("pandas,missing")


def test_target_descriptions_capture_methodology_axes():
    specs = describe_targets(["pandas", "duckdb"])
    assert [spec["family"] for spec in specs] == ["dataframe", "embedded_sql"]
    assert all(spec["adapter"] for spec in specs)
    assert "op:join" in specs[0]["capabilities"]
    assert "op:tuple_absence_filter" in specs[0]["capabilities"]
    assert "op:running_sum" in specs[0]["capabilities"]
    assert "op:sortedness_check" in specs[0]["capabilities"]
    assert "op:random_case_probe" in specs[0]["capabilities"]


def test_list_target_suites_includes_core():
    suites = {suite["suite"]: suite for suite in list_target_suites()}
    assert suites["core"]["backends"] == ["pandas", "polars", "duckdb", "sqlite"]
    assert suites["core_lazy"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["datafusion_cross"]["families"] == ["dataframe", "embedded_sql", "query_engine"]
    assert suites["latest_all_engines"]["families"] == ["arrow", "dataframe", "embedded_sql", "query_engine"]
    assert suites["cross_family"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["duckdb_storage_cross"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["dataframe_lazy"]["families"] == ["dataframe"]
    assert suites["seeded_groupby"]["families"] == ["dataframe", "seeded_fault"]
    assert "op:groupby" in suites["core"]["common_capabilities"]
    assert "op:running_sum" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:sortedness_check" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:random_case_probe" in suites["latest_all_engines"]["common_capabilities"]


def test_target_capability_matrix_and_intersection():
    matrix = target_capability_matrix(["pandas", "sqlite"])
    assert set(matrix) == {"pandas", "sqlite"}
    assert "expr:string_lower" in matrix["pandas"]
    assert "op:join" in common_capabilities(["pandas", "sqlite"])


def test_sqlite_target_records_runtime_version():
    env = collect_environment()

    assert SQLITE_RUNTIME in {"pysqlite3", "stdlib"}
    assert env["sqlite"] == SQLITE_VERSION
    assert env["sqlite_runtime"] == SQLITE_RUNTIME


def test_seeded_fault_targets_are_described():
    specs = describe_targets(["buggy_filter", "buggy_join"])
    assert [spec["family"] for spec in specs] == ["seeded_fault", "seeded_fault"]
    assert all(spec["layer"] == "fault_injection" for spec in specs)


def test_polars_lazy_target_is_described():
    spec = describe_targets(["polars_lazy"])[0]
    assert spec["family"] == "dataframe"
    assert spec["layer"] == "python_dataframe_lazy"


def test_datafusion_target_is_described():
    spec = describe_targets(["datafusion"])[0]
    assert spec["family"] == "query_engine"
    assert spec["layer"] == "arrow_query_engine"


def test_pyarrow_target_is_described():
    spec = describe_targets(["pyarrow"])[0]
    assert spec["family"] == "arrow"
    assert spec["layer"] == "arrow_compute"


def test_duckdb_persistent_target_is_described():
    spec = describe_targets(["duckdb_persistent"])[0]
    assert spec["family"] == "embedded_sql"
    assert spec["layer"] == "embedded_analytical_engine_storage"
    assert "op:offset" in spec["capabilities"]
