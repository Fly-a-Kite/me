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
    assert resolve_target_backends(target_suite="polars_cross") == ["pandas", "polars", "polars_lazy"]
    assert resolve_target_backends(target_suite="polars_streaming_cross") == ["polars_lazy", "polars_streaming"]
    assert resolve_target_backends(target_suite="embedded_sql") == ["duckdb", "sqlite"]
    assert resolve_target_backends(target_suite="embedded_sql_cross") == ["pandas", "duckdb", "sqlite"]
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
    assert resolve_target_backends(target_suite="latest_no_datafusion") == [
        "pandas",
        "pyarrow",
        "polars",
        "polars_lazy",
        "duckdb",
        "sqlite",
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
    assert "op:row_number_filter" in specs[0]["capabilities"]
    assert "op:running_sum" in specs[0]["capabilities"]
    assert "op:sortedness_check" in specs[0]["capabilities"]
    assert "op:random_case_probe" in specs[0]["capabilities"]
    assert "op:group_quantile_probe" in specs[0]["capabilities"]
    assert "op:scalar_subquery_probe" in specs[0]["capabilities"]
    assert "op:window_avg_probe" in specs[0]["capabilities"]
    assert "op:struct_distinct_probe" in specs[0]["capabilities"]
    assert "op:bit_compare_probe" in specs[0]["capabilities"]
    assert "op:round_even_probe" in specs[0]["capabilities"]
    assert "op:float_literal_precision_probe" in specs[0]["capabilities"]
    assert "op:timestamp_precision_filter_probe" in specs[0]["capabilities"]
    assert "op:series_rtruediv_probe" in specs[0]["capabilities"]
    assert "op:uint64_isin_probe" in specs[0]["capabilities"]
    assert "op:tuple_anti_null_probe" in specs[0]["capabilities"]
    assert "op:setop_all_duplicate_probe" in specs[0]["capabilities"]
    assert "op:json_predicate_order_probe" in specs[0]["capabilities"]
    assert "op:sparse_mask_probe" in specs[0]["capabilities"]
    assert "op:float_wrap_probe" in specs[0]["capabilities"]
    assert "op:index_bool_probe" in specs[0]["capabilities"]
    assert "op:empty_literal_groupby_probe" in specs[0]["capabilities"]
    assert "op:arrow_string_eq_sum_probe" in specs[0]["capabilities"]
    assert "op:arrow_timestamp_loc_slice_probe" in specs[0]["capabilities"]
    assert "op:arrow_timestamp_index_attr_probe" in specs[0]["capabilities"]
    assert "op:eval_inplace_alias_probe" in specs[0]["capabilities"]
    assert "op:bool_reduction_skipna_probe" in specs[0]["capabilities"]
    assert "op:dataset_isin_all_match_probe" in specs[0]["capabilities"]
    assert "op:run_end_null_compute_probe" in specs[0]["capabilities"]
    assert "op:large_string_partition_probe" in specs[0]["capabilities"]
    assert "op:hash_pivot_wider_probe" in specs[0]["capabilities"]
    assert "op:rolling_mean_by_null_count_probe" in specs[0]["capabilities"]
    assert "op:csv_long_numeric_roundtrip_probe" in specs[0]["capabilities"]
    assert "expr:string_basename" in specs[0]["capabilities"]
    assert "agg:any" in specs[0]["capabilities"]
    assert "agg:all" in specs[0]["capabilities"]
    assert "agg:nunique" in specs[0]["capabilities"]


def test_list_target_suites_includes_core():
    suites = {suite["suite"]: suite for suite in list_target_suites()}
    assert suites["core"]["backends"] == ["pandas", "polars", "duckdb", "sqlite"]
    assert suites["core_lazy"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["polars_cross"]["families"] == ["dataframe"]
    assert suites["embedded_sql_cross"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["datafusion_cross"]["families"] == ["dataframe", "embedded_sql", "query_engine"]
    assert suites["latest_all_engines"]["families"] == ["arrow", "dataframe", "embedded_sql", "query_engine"]
    assert suites["latest_no_datafusion"]["families"] == ["arrow", "dataframe", "embedded_sql"]
    assert suites["cross_family"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["duckdb_storage_cross"]["families"] == ["dataframe", "embedded_sql"]
    assert suites["dataframe_lazy"]["families"] == ["dataframe"]
    assert suites["seeded_groupby"]["families"] == ["dataframe", "seeded_fault"]
    assert "op:groupby" in suites["core"]["common_capabilities"]
    assert "op:running_sum" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:sortedness_check" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:random_case_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:group_quantile_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:scalar_subquery_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:window_avg_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:struct_distinct_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:bit_compare_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:round_even_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:float_literal_precision_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:timestamp_precision_filter_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:series_rtruediv_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:uint64_isin_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:tuple_anti_null_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:setop_all_duplicate_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:json_predicate_order_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:sparse_mask_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:float_wrap_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:index_bool_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:empty_literal_groupby_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:arrow_string_eq_sum_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:arrow_timestamp_loc_slice_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:arrow_timestamp_index_attr_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:eval_inplace_alias_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:bool_reduction_skipna_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:dataset_isin_all_match_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:run_end_null_compute_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:large_string_partition_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:hash_pivot_wider_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:rolling_mean_by_null_count_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:csv_long_numeric_roundtrip_probe" in suites["latest_all_engines"]["common_capabilities"]
    assert "op:row_number_filter" in suites["latest_all_engines"]["common_capabilities"]
    assert "expr:string_basename" in suites["latest_all_engines"]["common_capabilities"]
    assert "agg:any" in suites["latest_all_engines"]["common_capabilities"]
    assert "agg:all" in suites["latest_all_engines"]["common_capabilities"]


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


def test_polars_streaming_target_is_described():
    spec = describe_targets(["polars_streaming"])[0]
    assert spec["family"] == "dataframe"
    assert spec["layer"] == "python_dataframe_streaming"


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
