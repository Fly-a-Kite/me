from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
from typing import Any
from typing import Sequence


PROJECT_METHODOLOGY_NAME = "capability_aware_closed_loop_semantic_differential_fuzzing"
PROJECT_METHODOLOGY_ORIGIN = (
    "A target-agnostic method that keeps one typed semantic frontend, one normalized-result "
    "comparison core, and one evidence pipeline while swapping only the target registry, "
    "capability declaration, and backend adapter."
)
PROJECT_REUSABLE_LAYERS: tuple[str, ...] = (
    "typed_ir_semantic_frontend",
    "capability_aware_target_registry",
    "backend_adapter_lowering_boundary",
    "normalized_result_comparison_kernel",
    "differential_and_metamorphic_oracles",
    "classification_recheck_reduce_dedup_issue_pipeline",
    "feedback_guided_scheduler_and_reporting",
)
DEFAULT_TARGET_METHODOLOGY_ROLES: tuple[str, ...] = (
    "semantic_frontend_consumer",
    "capability_gated_suite_member",
    "normalized_result_participant",
    "differential_oracle_participant",
    "metamorphic_oracle_participant",
    "classification_and_evidence_participant",
)
DEFAULT_TARGET_EXTENSION_CONTRACT: tuple[str, ...] = (
    "declare_target_spec",
    "declare_capabilities",
    "implement_backend_adapter",
    "emit_backend_result",
    "reuse_normalizer_and_oracles",
    "compose_target_suite_and_preset",
)


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
    execution_model: str = "tabular_engine"
    result_contract: str = "normalized_result_v1"
    portability_tier: str = "adapter_reuse"
    methodology_roles: tuple[str, ...] = DEFAULT_TARGET_METHODOLOGY_ROLES
    extension_contract: tuple[str, ...] = DEFAULT_TARGET_EXTENSION_CONTRACT
    adapter_args: tuple[Any, ...] = ()
    adapter_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return _json_ready(asdict(self))

    def instantiate_backend(self) -> Any:
        module_name, sep, class_name = self.adapter.rpartition(".")
        if not sep or not module_name or not class_name:
            raise ValueError(f"target {self.backend} declares an invalid adapter path: {self.adapter}")
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name, None)
        if adapter_cls is None:
            raise ValueError(f"target {self.backend} adapter class is missing: {self.adapter}")
        return adapter_cls(*self.adapter_args, **self.adapter_kwargs)


@dataclass(frozen=True, slots=True)
class TargetContext:
    backends: tuple[str, ...]
    targets: tuple[TargetSpec, ...]
    families: tuple[str, ...]
    layers: tuple[str, ...]
    execution_models: tuple[str, ...]
    result_contracts: tuple[str, ...]
    portability_tiers: tuple[str, ...]
    common_capabilities: tuple[str, ...]
    shared_methodology_roles: tuple[str, ...]
    shared_extension_contract: tuple[str, ...]

    def target_dicts(self) -> list[dict[str, object]]:
        return [target.to_dict() for target in self.targets]

    def methodology_summary(self) -> dict[str, Any]:
        return {
            "name": PROJECT_METHODOLOGY_NAME,
            "origin": PROJECT_METHODOLOGY_ORIGIN,
            "reusable_layers": list(PROJECT_REUSABLE_LAYERS),
            "selected_backends": list(self.backends),
            "families": list(self.families),
            "layers": list(self.layers),
            "execution_models": list(self.execution_models),
            "result_contracts": list(self.result_contracts),
            "portability_tiers": list(self.portability_tiers),
            "shared_methodology_roles": list(self.shared_methodology_roles),
            "shared_extension_contract": list(self.shared_extension_contract),
            "shared_capabilities": list(self.common_capabilities),
            "extensibility_claim": (
                "Adding a new target should require a new TargetSpec declaration plus a backend adapter, "
                "while reusing the typed IR, normalization, oracle, scheduling, and evidence pipeline."
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "backends": list(self.backends),
            "families": list(self.families),
            "layers": list(self.layers),
            "execution_models": list(self.execution_models),
            "result_contracts": list(self.result_contracts),
            "portability_tiers": list(self.portability_tiers),
            "common_capabilities": list(self.common_capabilities),
            "shared_methodology_roles": list(self.shared_methodology_roles),
            "shared_extension_contract": list(self.shared_extension_contract),
            "targets": self.target_dicts(),
            "methodology": self.methodology_summary(),
        }


COMMON_DSL_CAPABILITIES: tuple[str, ...] = (
    "table:single",
    "table:multi",
    "op:filter",
    "filter:str_contains",
    "filter:str_starts_with",
    "filter:str_ends_with",
    "op:tuple_absence_filter",
    "op:drop_nulls",
    "op:union_all",
    "op:semi_join",
    "op:anti_join",
    "op:row_number_filter",
    "op:select",
    "op:distinct",
    "op:fill_null",
    "op:coalesce",
    "op:case_when",
    "op:sort",
    "op:limit",
    "op:offset",
    "op:mutate",
    "op:running_sum",
    "running:partition_by",
    "op:sortedness_check",
    "op:random_case_probe",
    "op:group_quantile_probe",
    "op:scalar_subquery_probe",
    "op:window_avg_probe",
    "op:struct_distinct_probe",
    "op:bit_compare_probe",
    "op:round_even_probe",
    "op:float_literal_precision_probe",
    "op:timestamp_precision_filter_probe",
    "op:series_rtruediv_probe",
    "op:uint64_isin_probe",
    "op:tuple_anti_null_probe",
    "op:setop_all_duplicate_probe",
    "op:json_predicate_order_probe",
    "op:sparse_mask_probe",
    "op:float_wrap_probe",
    "op:index_bool_probe",
    "op:empty_literal_groupby_probe",
    "op:arrow_string_eq_sum_probe",
    "op:arrow_timestamp_loc_slice_probe",
    "op:arrow_timestamp_index_attr_probe",
    "op:eval_inplace_alias_probe",
    "op:bool_reduction_skipna_probe",
    "op:arrow_bool_groupby_reduction_probe",
    "op:dataset_isin_all_match_probe",
    "op:run_end_null_compute_probe",
    "op:large_string_partition_probe",
    "op:hash_pivot_wider_probe",
    "op:list_flatten_parent_indices_probe",
    "op:rolling_mean_by_null_count_probe",
    "op:csv_long_numeric_roundtrip_probe",
    "op:groupby",
    "op:aggregate",
    "op:join",
    "expr:add_const",
    "expr:arith_const",
    "expr:abs",
    "expr:clip",
    "expr:bool_not",
    "expr:reverse_division_columns",
    "expr:cast",
    "expr:string_length",
    "expr:string_lower",
    "expr:string_upper",
    "expr:string_strip",
    "expr:string_null_if_empty",
    "expr:string_replace",
    "expr:string_slice",
    "expr:string_split_part",
    "expr:string_concat",
    "expr:string_contains",
    "expr:string_starts_with",
    "expr:string_ends_with",
    "expr:date_part",
    "expr:string_basename",
    "agg:sum",
    "agg:mean",
    "agg:min",
    "agg:max",
    "agg:count",
    "agg:nunique",
    "agg:any",
    "agg:all",
    "type:int",
    "type:float",
    "type:bool",
    "type:str",
    "nulls",
)


# chDB v1: SQL frontend without native running_sum / row_number_filter /
# sortedness_check / scalar_subquery_probe. The backend stubs probe-kinds
# in EXTENDED_FALSE_PROBE_KINDS the same way SQLite does, so those caps
# remain advertised. Ops below are dropped to avoid routing cases that
# would resolve to BackendResult(status="missing").
_CHDB_DISABLED_CAPS: frozenset[str] = frozenset(
    {
        "op:running_sum",
        "running:partition_by",
        "op:row_number_filter",
        "op:sortedness_check",
        "op:scalar_subquery_probe",
    }
)
CHDB_DSL_CAPABILITIES: tuple[str, ...] = tuple(
    cap for cap in COMMON_DSL_CAPABILITIES if cap not in _CHDB_DISABLED_CAPS
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
        execution_model="dataframe_api",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="dataframe_api",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="lazy_dataframe_plan",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
    ),
    "polars_streaming": TargetSpec(
        name="polars_streaming",
        backend="polars_streaming",
        family="dataframe",
        layer="python_dataframe_streaming",
        adapter="datadiff.backends.polars_backend.PolarsStreamingBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Polars lazy streaming engine execution path for streaming optimizer-sensitive differential tests.",
        execution_model="streaming_dataframe_plan",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="embedded_sql_engine",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="embedded_sql_storage_engine",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="arrow_query_engine",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="arrow_table_compute",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
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
        execution_model="embedded_sql_engine",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
    ),
    "chdb": TargetSpec(
        name="chdb",
        backend="chdb",
        family="embedded_olap",
        layer="embedded_columnar_engine",
        adapter="datadiff.backends.chdb_backend.ChDBBackend",
        status="experimental",
        capabilities=CHDB_DSL_CAPABILITIES,
        description="Embedded ClickHouse (chDB) columnar OLAP engine — distinct optimizer family from DuckDB/SQLite.",
        execution_model="embedded_columnar_olap",
        portability_tier="optional_dependency",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
    ),
    "buggy_filter": TargetSpec(
        name="buggy_filter",
        backend="buggy_filter",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected filter output fault for evaluation.",
        execution_model="fault_injection_wrapper",
        portability_tier="seeded_fault_reuse",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
        adapter_args=("buggy_filter", "filter"),
    ),
    "buggy_groupby": TargetSpec(
        name="buggy_groupby",
        backend="buggy_groupby",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected groupby aggregate fault for evaluation.",
        execution_model="fault_injection_wrapper",
        portability_tier="seeded_fault_reuse",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
        adapter_args=("buggy_groupby", "groupby"),
    ),
    "buggy_join": TargetSpec(
        name="buggy_join",
        backend="buggy_join",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected join cardinality fault for evaluation.",
        execution_model="fault_injection_wrapper",
        portability_tier="seeded_fault_reuse",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
        adapter_args=("buggy_join", "join"),
    ),
    "buggy_mutate": TargetSpec(
        name="buggy_mutate",
        backend="buggy_mutate",
        family="seeded_fault",
        layer="fault_injection",
        adapter="datadiff.backends.faulty_backend.FaultyPandasBackend",
        status="implemented",
        capabilities=COMMON_DSL_CAPABILITIES,
        description="Pandas-compatible backend with an injected mutate expression fault for evaluation.",
        execution_model="fault_injection_wrapper",
        portability_tier="seeded_fault_reuse",
        methodology_roles=DEFAULT_TARGET_METHODOLOGY_ROLES,
        extension_contract=DEFAULT_TARGET_EXTENSION_CONTRACT,
        adapter_args=("buggy_mutate", "mutate"),
    ),
}

TARGET_SUITES: dict[str, list[str]] = {
    "dataframe": ["pandas", "polars"],
    "dataframe_lazy": ["polars", "polars_lazy"],
    "polars_cross": ["pandas", "polars", "polars_lazy"],
    "polars_streaming_cross": ["polars_lazy", "polars_streaming"],
    "embedded_sql": ["duckdb", "sqlite"],
    "embedded_sql_cross": ["pandas", "duckdb", "sqlite"],
    "duckdb_storage_cross": ["pandas", "duckdb_persistent"],
    "cross_family": ["pandas", "duckdb"],
    "lazy_cross_family": ["pandas", "polars_lazy", "duckdb"],
    "core_lazy": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite"],
    "datafusion_cross": ["pandas", "duckdb", "datafusion"],
    "core_datafusion": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"],
    "arrow_cross": ["pandas", "duckdb", "pyarrow"],
    "core_arrow": ["pandas", "polars", "polars_lazy", "duckdb", "sqlite", "pyarrow"],
    "latest_all_engines": ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "datafusion"],
    "latest_no_datafusion": ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite"],
    "chdb_cross": ["pandas", "duckdb", "chdb"],
    "chdb_olap_cross": ["pandas", "duckdb", "sqlite", "chdb"],
    "latest_with_chdb": ["pandas", "pyarrow", "polars", "polars_lazy", "duckdb", "sqlite", "chdb"],
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


def target_spec(backend: str) -> TargetSpec:
    if backend not in TARGETS:
        raise ValueError(f"unknown target backend: {backend}")
    return TARGETS[backend]


def instantiate_target_backend(backend: str) -> Any:
    return target_spec(backend).instantiate_backend()


def target_specs(backends: Sequence[str]) -> list[TargetSpec]:
    return [target_spec(backend) for backend in backends]


def target_context(
    backends: Sequence[str] | None = None,
    *,
    target_suite: str | None = None,
) -> TargetContext:
    selected = (
        resolve_target_backends(target_suite=target_suite or "core")
        if backends is None
        else resolve_target_backends(list(backends), target_suite=target_suite or "core")
    )
    specs = tuple(target_specs(selected))
    families = tuple(sorted({spec.family for spec in specs}))
    layers = tuple(sorted({spec.layer for spec in specs}))
    execution_models = tuple(sorted({spec.execution_model for spec in specs}))
    result_contracts = tuple(sorted({spec.result_contract for spec in specs}))
    portability_tiers = tuple(sorted({spec.portability_tier for spec in specs}))
    shared_caps = tuple(common_capabilities(list(selected)))
    shared_roles = tuple(_common_string_contract(list(specs), attr="methodology_roles"))
    shared_extension_contract = tuple(_common_string_contract(list(specs), attr="extension_contract"))
    return TargetContext(
        backends=tuple(selected),
        targets=specs,
        families=families,
        layers=layers,
        execution_models=execution_models,
        result_contracts=result_contracts,
        portability_tiers=portability_tiers,
        common_capabilities=shared_caps,
        shared_methodology_roles=shared_roles,
        shared_extension_contract=shared_extension_contract,
    )


def describe_targets(backends: list[str]) -> list[dict[str, object]]:
    return target_context(backends).target_dicts()


def list_target_suites() -> list[dict[str, str | list[str]]]:
    suites: list[dict[str, str | list[str]]] = []
    for suite, backends in sorted(TARGET_SUITES.items()):
        context = target_context(backends, target_suite=suite)
        suites.append(
            {
                "suite": suite,
                "backends": list(context.backends),
                "families": list(context.families),
                "layers": list(context.layers),
                "execution_models": list(context.execution_models),
                "common_capabilities": list(context.common_capabilities),
            }
        )
    return suites


def target_capability_matrix(backends: list[str] | None = None) -> dict[str, list[str]]:
    selected = list(backends) if backends is not None else sorted(TARGETS)
    return {
        backend: list(target_spec(backend).capabilities)
        for backend in selected
        if backend in TARGETS
    }


def common_capabilities(backends: list[str]) -> list[str]:
    selected = [target_spec(backend) for backend in backends if backend in TARGETS]
    if not selected:
        return []
    common = set(selected[0].capabilities)
    for target in selected[1:]:
        common &= set(target.capabilities)
    return sorted(common)


def describe_methodology(backends: list[str] | None = None) -> dict[str, Any]:
    context = target_context(backends or sorted(TARGETS))
    return context.methodology_summary()


def validate_target_registry() -> None:
    for backend, spec in TARGETS.items():
        if spec.name != backend or spec.backend != backend:
            raise ValueError(f"target spec identity mismatch for {backend}")
        if not spec.capabilities:
            raise ValueError(f"target {backend} must declare capabilities")
        if not spec.methodology_roles:
            raise ValueError(f"target {backend} must declare methodology roles")
        if not spec.extension_contract:
            raise ValueError(f"target {backend} must declare an extension contract")
        if not spec.adapter:
            raise ValueError(f"target {backend} must declare an adapter")
        if "(" in spec.adapter or ")" in spec.adapter:
            raise ValueError(
                f"target {backend} must declare adapter constructor arguments separately from the adapter path"
            )
        if not spec.adapter.rpartition(".")[1]:
            raise ValueError(f"target {backend} adapter must be a fully-qualified class path: {spec.adapter}")
    for suite, backends in TARGET_SUITES.items():
        if not backends:
            raise ValueError(f"target suite {suite} must not be empty")
        unknown = [backend for backend in backends if backend not in TARGETS]
        if unknown:
            raise ValueError(f"target suite {suite} references unknown backends: {', '.join(unknown)}")


def _common_string_contract(targets: list[TargetSpec], *, attr: str) -> list[str]:
    if not targets:
        return []
    common = set(getattr(targets[0], attr))
    for target in targets[1:]:
        common &= set(getattr(target, attr))
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


validate_target_registry()
