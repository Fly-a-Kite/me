from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TargetSpec:
    name: str
    backend: str
    family: str
    layer: str
    adapter: str
    status: str
    capabilities: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


COMMON_DSL_CAPABILITIES: tuple[str, ...] = (
    "table:single",
    "table:multi",
    "op:filter",
    "op:tuple_absence_filter",
    "op:select",
    "op:sort",
    "op:limit",
    "op:offset",
    "op:mutate",
    "op:running_sum",
    "op:sortedness_check",
    "op:random_case_probe",
    "op:group_quantile_probe",
    "op:scalar_subquery_probe",
    "op:window_avg_probe",
    "op:struct_distinct_probe",
    "op:bit_compare_probe",
    "op:round_even_probe",
    "op:series_rtruediv_probe",
    "op:uint64_isin_probe",
    "op:tuple_anti_null_probe",
    "op:sparse_mask_probe",
    "op:float_wrap_probe",
    "op:index_bool_probe",
    "op:empty_literal_groupby_probe",
    "op:arrow_string_eq_sum_probe",
    "op:arrow_timestamp_loc_slice_probe",
    "op:groupby",
    "op:aggregate",
    "op:join",
    "expr:add_const",
    "expr:arith_const",
    "expr:cast",
    "expr:string_length",
    "expr:string_lower",
    "agg:sum",
    "agg:min",
    "agg:max",
    "agg:count",
    "type:int",
    "type:float",
    "type:bool",
    "type:str",
    "nulls",
)


TARGETS: dict[str, TargetSpec] = {
    "pandas": TargetSpec(
        name="pandas",
        backend="pandas",
        family="dataframe",
        layer="python_dataframe",
        adapter="datadiff.backends.pandas_backend.PandasBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Python DataFrame baseline with pandas semantics.",
    ),
    "polars": TargetSpec(
        name="polars",
        backend="polars",
        family="dataframe",
        layer="python_dataframe",
        adapter="datadiff.backends.polars_backend.PolarsBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Columnar DataFrame engine with eager Polars semantics.",
    ),
    "polars_lazy": TargetSpec(
        name="polars_lazy",
        backend="polars_lazy",
        family="dataframe",
        layer="python_dataframe_lazy",
        adapter="datadiff.backends.polars_backend.PolarsLazyBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Polars lazy query execution path for optimizer-sensitive differential tests.",
    ),
    "duckdb": TargetSpec(
        name="duckdb",
        backend="duckdb",
        family="embedded_sql",
        layer="embedded_analytical_engine",
        adapter="datadiff.backends.duckdb_backend.DuckDBBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Embedded analytical SQL engine executed over generated tables.",
    ),
    "duckdb_persistent": TargetSpec(
        name="duckdb_persistent",
        backend="duckdb_persistent",
        family="embedded_sql",
        layer="embedded_analytical_engine_storage",
        adapter="datadiff.backends.duckdb_backend.DuckDBPersistentBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="DuckDB executed over a temporary persisted database to exercise storage-aware optimizer paths.",
    ),
    "datafusion": TargetSpec(
        name="datafusion",
        backend="datafusion",
        family="query_engine",
        layer="arrow_query_engine",
        adapter="datadiff.backends.datafusion_backend.DataFusionBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Apache DataFusion SQL engine over Arrow record batches.",
    ),
    "pyarrow": TargetSpec(
        name="pyarrow",
        backend="pyarrow",
        family="arrow",
        layer="arrow_compute",
        adapter="datadiff.backends.pyarrow_backend.PyArrowBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Apache Arrow table/compute backend using PyArrow kernels.",
    ),
    "sqlite": TargetSpec(
        name="sqlite",
        backend="sqlite",
        family="embedded_sql",
        layer="embedded_sql_engine",
        adapter="datadiff.backends.sqlite_backend.SQLiteBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Embedded SQL reference target for common relational operators.",
    ),
    "buggy_filter": TargetSpec(
        name="buggy_filter",
        backend="buggy_filter",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend(filter)",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected filter output fault for evaluation.",
    ),
    "buggy_groupby": TargetSpec(
        name="buggy_groupby",
        backend="buggy_groupby",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend(groupby)",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected groupby aggregate fault for evaluation.",
    ),
    "buggy_join": TargetSpec(
        name="buggy_join",
        backend="buggy_join",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend(join)",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected join cardinality fault for evaluation.",
    ),
    "buggy_mutate": TargetSpec(
        name="buggy_mutate",
        backend="buggy_mutate",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend(mutate)",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected mutate expression fault for evaluation.",
    ),
}

TARGET_SUITES: dict[str, list[str]] = {
    "dataframe": ["pandas", "polars"],
    "dataframe_lazy": ["polars", "polars_lazy"],
    "embedded_sql": ["duckdb", "sqlite"],
    "duckdb_storage_cross": ["pandas", "duckdb_persistent"],
    "cross_family": ["pandas", "duckdb"],
    "lazy_cross_family": ["pandas", "polars_lazy", "duckdb"],
    "core_lazy": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite"],
    "datafusion_cross": ["pandas", "duckdb", "datafusion"],
    "core_datafusion": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"],
    "arrow_cross": ["pandas", "duckdb", "pyarrow"],
    "core_arrow": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow"],
    "latest_all_engines": ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"],
    "seeded_filter": ["pandas", "buggy_filter"],
    "seeded_groupby": ["pandas", "buggy_groupby"],
    "seeded_join": ["pandas", "buggy_join"],
    "seeded_mutate": ["pandas", "buggy_mutate"],
    "core": ["pandas", "polars", "duckdb", "sqlite"],
    "all": ["pandas", "polars", "duckdb", "sqlite"],
}


def parse_backend_names(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, (list, tuple)) else value.split(",")
    backends = []
    for item in raw:
        name = str(item).strip()
        if name:
            backends.append(name)
    return _dedupe(backends)


def resolve_target_backends(
    backends: str | list[str] | tuple[str, ...] | None = None,
    target_suite: str = "core",
) -> list[str]:
    explicit = parse_backend_names(backends)
    selected = explicit or list(TARGET_SUITES.get(target_suite, []))
    if not selected:
        raise ValueError(f"unknown target suite: {target_suite}")
    unknown = [backend for backend in selected if backend not in TARGETS]
    if unknown:
        raise ValueError(f"unknown target backend(s): {', '.join(unknown)}")
    return selected


def describe_targets(backends: list[str]) -> list[dict[str, object]]:
    return [TARGETS[backend].to_dict() for backend in backends if backend in TARGETS]


def list_target_suites() -> list[dict[str, str | list[str]]]:
    return [
        {
            "suite": suite,
            "backends": list(backends),
            "families": sorted({TARGETS[backend].family for backend in backends if backend in TARGETS}),
            "common_capabilities": common_capabilities(backends),
        }
        for suite, backends in sorted(TARGET_SUITES.items())
    ]


def target_capability_matrix(backends: list[str] | None = None) -> dict[str, list[str]]:
    selected = backends or sorted(TARGETS)
    return {
        backend: list(TARGETS[backend].capabilities)
        for backend in selected
        if backend in TARGETS
    }


def common_capabilities(backends: list[str]) -> list[str]:
    selected = [TARGETS[backend] for backend in backends if backend in TARGETS]
    if not selected:
        return []
    common = set(selected[0].capabilities)
    for target in selected[1:]:
        common &= set(target.capabilities)
    return sorted(common)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
