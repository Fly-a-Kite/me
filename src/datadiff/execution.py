from __future__ import annotations

from collections.abc import Callable
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from datadiff.backends import make_backend
from datadiff.backends.base import Backend, BackendResult, prepare_tables
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, Program
from datadiff.execution_cache import CachedExecutionRow, ExecutionResultCache
from datadiff.family_witness_registry import family_witness_screening_plan_policy
from datadiff.ir_runtime import resolve_case_program
from datadiff.normalizer import normalize_result
from datadiff.osc_diagnostic_facade import build_legacy_diagnostic_ref_set
from datadiff.physical_layout import (
    PhysicalLayoutResolution,
    resolve_case_physical_layout,
)
from datadiff.targets import TARGETS, target_spec

BackendFactory = Callable[[str], Backend]


class BackendIsolationError(RuntimeError):
    pass


def physical_plan_detail_for_case_backend(
    case: Case,
    backend: Backend,
    config: ExperimentConfig,
) -> str:
    """Resolve plan detail from method, evidence tier, family policy, and SPI."""

    detail = config.method_policy.execution.plan_collection.detail_for(
        config.evidence_tier
    )
    if (
        config.evidence_tier == "screening"
        and family_witness_screening_plan_policy(case) == "finding_only"
    ):
        detail = "disabled"
    if backend.plan_collection_support == "unsupported":
        detail = "disabled"
    return detail


class BackendExecutor:
    def __init__(
        self,
        backend_names: list[str],
        *,
        backend_instances: dict[str, Backend] | None = None,
        backend_factory: BackendFactory = make_backend,
        allow_backend_factory_fallback: bool = True,
        enforce_session_isolation: bool = False,
    ) -> None:
        self.backend_names = list(backend_names)
        self._backend_instances = dict(backend_instances or {})
        self._backend_factory = backend_factory
        self._allow_backend_factory_fallback = allow_backend_factory_fallback
        self._enforce_session_isolation = bool(enforce_session_isolation)
        self.session_reset_calls = 0
        self.session_isolation_rejections = 0

    @property
    def backend_instances(self) -> dict[str, Backend]:
        return self._backend_instances

    def execute(
        self,
        case: Case,
        config: ExperimentConfig,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        raw_results: dict[str, dict[str, Any]] = {}
        normalized: dict[str, Any] = {}
        resolved = resolve_case_program(case, config)
        prepared_tables = prepare_tables(case.tables)
        required_capabilities = (
            resolved.ccs_ir.required_capabilities if resolved.ccs_ir is not None else ()
        )
        for backend_name in self.backend_names:
            backend = self._backend_for_name(backend_name)
            raw_row, normalized_row = self._execute_one(
                backend_name,
                prepared_tables,
                resolved.program,
                config,
                required_capabilities=required_capabilities,
                physical_layout=resolve_case_physical_layout(
                    case,
                    backend_name,
                    backend,
                ),
                plan_detail=physical_plan_detail_for_case_backend(
                    case,
                    backend,
                    config,
                ),
            )
            raw_results[backend_name] = raw_row
            normalized[backend_name] = normalized_row
        return raw_results, normalized

    def execute_batch(
        self,
        cases: list[Case],
        config: ExperimentConfig,
    ) -> list[tuple[dict[str, dict[str, Any]], dict[str, Any]]]:
        return [self.execute(case, config) for case in cases]

    def _execute_one(
        self,
        backend_name: str,
        prepared_tables: list[Any],
        program: Program,
        config: ExperimentConfig,
        *,
        required_capabilities: tuple[str, ...] = (),
        physical_layout: PhysicalLayoutResolution,
        plan_detail: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        backend = self._backend_for_name(backend_name)
        self._prepare_reused_backend(backend_name, backend)
        previous_layout = str(backend.input_physical_layout or "default")
        try:
            try:
                backend.configure_physical_layout(physical_layout.applied_layout)
                resolved_plan_detail = (
                    config.method_policy.execution.plan_collection.detail_for(
                        config.evidence_tier
                    )
                    if plan_detail is None
                    else plan_detail
                )
                if backend.plan_collection_support == "unsupported":
                    resolved_plan_detail = "disabled"
                backend.configure_physical_plan_collection(resolved_plan_detail)
                capability_decision = None
                if backend_name in TARGETS:
                    capability_decision = target_spec(
                        backend_name
                    ).capability_decision(
                        required_tokens=required_capabilities,
                        physical_layout=physical_layout.applied_layout,
                    ).to_dict()
                if (
                    capability_decision is not None
                    and not capability_decision["supported"]
                ):
                    result = BackendResult(
                        backend=backend_name,
                        status="missing",
                        error_type="UnsupportedCapability",
                        error=str(capability_decision["skip_reason"]),
                        capability_decision=capability_decision,
                    )
                else:
                    result = backend.run(prepared_tables, program)
                result.capability_decision = capability_decision
            except Exception as exc:  # noqa: BLE001
                result = BackendResult(
                    backend=backend_name,
                    status="error",
                    error_type=f"AdapterBoundaryError:{type(exc).__name__}",
                    error=str(exc),
                )
            raw_row = result.summary_dict()
            raw_row["input_physical_layout"] = physical_layout.applied_layout
            raw_row["physical_layout_resolution"] = physical_layout.to_dict()
            normalized_row = normalize_result(
                result,
                program,
                enable_normalizer=config.enable_normalizer,
            )
        finally:
            backend.configure_physical_layout(previous_layout)
        return raw_row, normalized_row

    def _prepare_reused_backend(self, backend_name: str, backend: Backend) -> None:
        if not self._enforce_session_isolation:
            return
        policy = str(backend.session_reuse_policy or "stateless")
        if policy == "stateless":
            return
        if policy == "reset":
            self.session_reset_calls += 1
            if not bool(
                getattr(backend, "session_reset_managed_by_backend", False)
            ):
                backend.reset_for_case()
            return
        self.session_isolation_rejections += 1
        raise BackendIsolationError(
            f"backend {backend_name} requires fresh instances (policy={policy})"
        )

    def _backend_for_name(self, backend_name: str) -> Backend:
        backend = self._backend_instances.get(backend_name)
        if backend is None:
            if not self._allow_backend_factory_fallback:
                return self._backend_instances[backend_name]
            backend = self._backend_factory(backend_name)
            self._backend_instances[backend_name] = backend
        return backend


class ParallelBackendExecutor(BackendExecutor):
    def __init__(
        self,
        backend_names: list[str],
        *,
        backend_instances: dict[str, Backend] | None = None,
        backend_factory: BackendFactory = make_backend,
        allow_backend_factory_fallback: bool = True,
        max_workers: int | None = None,
        thread_pool: ThreadPoolExecutor | None = None,
        enforce_session_isolation: bool = False,
    ) -> None:
        super().__init__(
            backend_names,
            backend_instances=backend_instances,
            backend_factory=backend_factory,
            allow_backend_factory_fallback=allow_backend_factory_fallback,
            enforce_session_isolation=enforce_session_isolation,
        )
        self._max_workers = max_workers
        self._thread_pool = thread_pool

    def execute(
        self,
        case: Case,
        config: ExperimentConfig,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if len(self.backend_names) <= 1 or len(set(self.backend_names)) != len(self.backend_names):
            return super().execute(case, config)

        physical_layouts: dict[str, PhysicalLayoutResolution] = {}
        plan_details: dict[str, str] = {}
        for backend_name in self.backend_names:
            backend = self._backend_for_name(backend_name)
            physical_layouts[backend_name] = resolve_case_physical_layout(
                case,
                backend_name,
                backend,
            )
            plan_details[backend_name] = physical_plan_detail_for_case_backend(
                case,
                backend,
                config,
            )

        resolved = resolve_case_program(case, config)
        prepared_tables = prepare_tables(case.tables)
        required_capabilities = (
            resolved.ccs_ir.required_capabilities if resolved.ccs_ir is not None else ()
        )
        worker_count = max(1, min(len(self.backend_names), int(self._max_workers or len(self.backend_names))))
        owned_pool = None
        pool = self._thread_pool
        if pool is None:
            owned_pool = ThreadPoolExecutor(max_workers=worker_count)
            pool = owned_pool
        try:
            futures = {
                backend_name: pool.submit(
                    self._execute_one,
                    backend_name,
                    prepared_tables,
                    resolved.program,
                    config,
                    required_capabilities=required_capabilities,
                    physical_layout=physical_layouts[backend_name],
                    plan_detail=plan_details[backend_name],
                )
                for backend_name in self.backend_names
            }
            rows = {
                backend_name: future.result()
                for backend_name, future in futures.items()
            }
        finally:
            if owned_pool is not None:
                owned_pool.shutdown(wait=True)

        raw_results: dict[str, dict[str, Any]] = {}
        normalized: dict[str, Any] = {}
        for backend_name in self.backend_names:
            raw_row, normalized_row = rows[backend_name]
            raw_results[backend_name] = raw_row
            normalized[backend_name] = normalized_row
        return raw_results, normalized

    def execute_batch(
        self,
        cases: list[Case],
        config: ExperimentConfig,
    ) -> list[tuple[dict[str, dict[str, Any]], dict[str, Any]]]:
        if not cases:
            return []
        if len(self.backend_names) <= 1 or len(set(self.backend_names)) != len(self.backend_names):
            return super().execute_batch(cases, config)

        for backend_name in self.backend_names:
            self._backend_for_name(backend_name)
        prepared_cases = []
        for case in cases:
            resolved = resolve_case_program(case, config)
            physical_layouts = {
                backend_name: resolve_case_physical_layout(
                    case,
                    backend_name,
                    self._backend_for_name(backend_name),
                )
                for backend_name in self.backend_names
            }
            plan_details = {
                backend_name: physical_plan_detail_for_case_backend(
                    case,
                    self._backend_for_name(backend_name),
                    config,
                )
                for backend_name in self.backend_names
            }
            prepared_cases.append(
                (
                    prepare_tables(case.tables),
                    resolved.program,
                    resolved.ccs_ir.required_capabilities
                    if resolved.ccs_ir is not None
                    else (),
                    physical_layouts,
                    plan_details,
                )
            )
        worker_count = max(1, min(len(self.backend_names), int(self._max_workers or len(self.backend_names))))
        owned_pool = None
        pool = self._thread_pool
        if pool is None:
            owned_pool = ThreadPoolExecutor(max_workers=worker_count)
            pool = owned_pool
        try:
            futures = {
                backend_name: pool.submit(
                    self._execute_backend_batch,
                    backend_name,
                    prepared_cases,
                    config,
                )
                for backend_name in self.backend_names
            }
            rows_by_backend = {
                backend_name: future.result()
                for backend_name, future in futures.items()
            }
        finally:
            if owned_pool is not None:
                owned_pool.shutdown(wait=True)

        results: list[tuple[dict[str, dict[str, Any]], dict[str, Any]]] = []
        for case_index in range(len(cases)):
            raw_results: dict[str, dict[str, Any]] = {}
            normalized: dict[str, Any] = {}
            for backend_name in self.backend_names:
                raw_row, normalized_row = rows_by_backend[backend_name][case_index]
                raw_results[backend_name] = raw_row
                normalized[backend_name] = normalized_row
            results.append((raw_results, normalized))
        return results

    def _execute_backend_batch(
        self,
        backend_name: str,
        prepared_cases: list[
            tuple[
                list[Any],
                Program,
                tuple[str, ...],
                dict[str, PhysicalLayoutResolution],
                dict[str, str],
            ]
        ],
        config: ExperimentConfig,
    ) -> list[tuple[dict[str, Any], Any]]:
        return [
            self._execute_one(
                backend_name,
                prepared_tables,
                program,
                config,
                required_capabilities=required_capabilities,
                physical_layout=physical_layouts[backend_name],
                plan_detail=plan_details[backend_name],
            )
            for (
                prepared_tables,
                program,
                required_capabilities,
                physical_layouts,
                plan_details,
            ) in prepared_cases
        ]


class BackendExecutionSession:
    """Reuse backend instances and one worker pool across a sequential fuzz run."""

    def __init__(
        self,
        backend_names: list[str],
        *,
        backend_instances: dict[str, Backend] | None = None,
        backend_factory: BackendFactory = make_backend,
        max_workers: int | None = None,
        environment: Mapping[str, Any] | None = None,
        adapter_revision: str = "",
        cache_limit: int = 4096,
        cache_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.backend_names = list(backend_names)
        self.backend_instances = dict(backend_instances or {})
        self._backend_factory = backend_factory
        for backend_name in self.backend_names:
            if backend_name not in self.backend_instances:
                self.backend_instances[backend_name] = self._backend_factory(backend_name)
        self._max_workers = max(
            1,
            min(len(self.backend_names) or 1, int(max_workers or len(self.backend_names) or 1)),
        )
        self._thread_pool = (
            ThreadPoolExecutor(max_workers=self._max_workers)
            if len(self.backend_names) > 1 and len(set(self.backend_names)) == len(self.backend_names)
            else None
        )
        self._executors: dict[tuple[tuple[str, ...], bool], BackendExecutor] = {}
        self._cache = ExecutionResultCache(
            self.backend_instances,
            environment=environment,
            adapter_revision=adapter_revision,
            limit=cache_limit,
            max_bytes=cache_max_bytes,
        )
        self._closed = False
        self._adapter_close_errors: list[dict[str, str]] = []
        self.execute_calls = 0
        self.backend_calls = 0
        self.batch_calls = 0

    def execute_case(
        self,
        case: Case,
        backends: list[str],
        config: ExperimentConfig,
        backend_instances: dict[str, Backend] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("backend execution session is closed")
        self.execute_calls += 1
        if not self._cache.enabled_for(config):
            names, executor = self._executor_for(backends, config)
            self.backend_calls += len(names)
            return executor.execute(case, config)

        names = tuple(backends)
        self._validate_backend_names(names)
        case_digest = self._cache.case_digest(case, config)
        cached_rows: dict[str, CachedExecutionRow] = {}
        missing: list[str] = []
        cache_keys: dict[str, str] = {}
        for backend_name in names:
            backend = self.backend_instances[backend_name]
            physical_layout = resolve_case_physical_layout(
                case,
                backend_name,
                backend,
            )
            cache_key = self._cache.key(
                case_digest=case_digest,
                backend_name=backend_name,
                config=config,
                input_physical_layout=physical_layout.applied_layout,
                physical_plan_detail=physical_plan_detail_for_case_backend(
                    case,
                    backend,
                    config,
                ),
            )
            cache_keys[backend_name] = cache_key
            cached = self._cache.get(cache_key)
            if cached is None:
                missing.append(backend_name)
            else:
                cached_rows[backend_name] = cached
        executed_rows: dict[str, CachedExecutionRow] = {}
        if missing:
            missing_names, executor = self._executor_for(missing, config)
            self.backend_calls += len(missing_names)
            raw_results, normalized = executor.execute(case, config)
            for backend_name in missing_names:
                row = (raw_results[backend_name], normalized[backend_name])
                executed_rows[backend_name] = row
                self._cache.put(cache_keys[backend_name], row)
        return self._assemble_cached_result(names, cached_rows, executed_rows)

    def execute_cases(
        self,
        cases: list[Case],
        backends: list[str],
        config: ExperimentConfig,
        backend_instances: dict[str, Backend] | None = None,
    ) -> list[tuple[dict[str, dict[str, Any]], dict[str, Any]]]:
        if self._closed:
            raise RuntimeError("backend execution session is closed")
        resolved_cases = list(cases)
        if not resolved_cases:
            return []
        self.batch_calls += 1
        self.execute_calls += len(resolved_cases)
        if not self._cache.enabled_for(config):
            names, executor = self._executor_for(backends, config)
            self.backend_calls += len(names) * len(resolved_cases)
            return executor.execute_batch(resolved_cases, config)

        names = tuple(backends)
        self._validate_backend_names(names)
        cached_by_case: list[dict[str, CachedExecutionRow]] = []
        cache_keys_by_case: list[dict[str, str]] = []
        missing_groups: dict[tuple[str, ...], list[int]] = {}
        for case_index, case in enumerate(resolved_cases):
            case_digest = self._cache.case_digest(case, config)
            cached_rows: dict[str, CachedExecutionRow] = {}
            cache_keys: dict[str, str] = {}
            missing: list[str] = []
            for backend_name in names:
                backend = self.backend_instances[backend_name]
                physical_layout = resolve_case_physical_layout(
                    case,
                    backend_name,
                    backend,
                )
                cache_key = self._cache.key(
                    case_digest=case_digest,
                    backend_name=backend_name,
                    config=config,
                    input_physical_layout=physical_layout.applied_layout,
                    physical_plan_detail=physical_plan_detail_for_case_backend(
                        case,
                        backend,
                        config,
                    ),
                )
                cache_keys[backend_name] = cache_key
                cached = self._cache.get(cache_key)
                if cached is None:
                    missing.append(backend_name)
                else:
                    cached_rows[backend_name] = cached
            cached_by_case.append(cached_rows)
            cache_keys_by_case.append(cache_keys)
            missing_groups.setdefault(tuple(missing), []).append(case_index)

        executed_by_case: list[dict[str, CachedExecutionRow]] = [
            {} for _case in resolved_cases
        ]
        for missing_names, case_indices in missing_groups.items():
            if not missing_names:
                continue
            _resolved_names, executor = self._executor_for(list(missing_names), config)
            group_cases = [resolved_cases[index] for index in case_indices]
            self.backend_calls += len(missing_names) * len(group_cases)
            group_results = executor.execute_batch(group_cases, config)
            for case_index, (raw_results, normalized) in zip(
                case_indices,
                group_results,
                strict=True,
            ):
                for backend_name in missing_names:
                    row = (raw_results[backend_name], normalized[backend_name])
                    executed_by_case[case_index][backend_name] = row
                    self._cache.put(cache_keys_by_case[case_index][backend_name], row)

        return [
            self._assemble_cached_result(
                names,
                cached_by_case[index],
                executed_by_case[index],
            )
            for index in range(len(resolved_cases))
        ]

    def _validate_backend_names(self, names: tuple[str, ...]) -> None:
        unknown = [name for name in names if name not in self.backend_names]
        if unknown:
            raise KeyError(unknown[0])

    @staticmethod
    def _assemble_cached_result(
        names: tuple[str, ...],
        cached_rows: dict[str, CachedExecutionRow],
        executed_rows: dict[str, CachedExecutionRow],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        rows = {**cached_rows, **executed_rows}
        return (
            {backend_name: rows[backend_name][0] for backend_name in names},
            {backend_name: rows[backend_name][1] for backend_name in names},
        )

    def _executor_for(
        self,
        backends: list[str],
        config: ExperimentConfig,
    ) -> tuple[tuple[str, ...], BackendExecutor]:
        names = tuple(backends)
        unknown = [name for name in names if name not in self.backend_names]
        if unknown:
            raise KeyError(unknown[0])
        use_parallel = (
            bool(getattr(config, "enable_parallel_backend_execution", True))
            and len(names) > 1
            and len(set(names)) == len(names)
        )
        key = (names, use_parallel)
        executor = self._executors.get(key)
        if executor is None:
            common_kwargs = {
                "backend_instances": self.backend_instances,
                "backend_factory": self._backend_factory,
                "allow_backend_factory_fallback": True,
                "enforce_session_isolation": True,
            }
            if use_parallel:
                executor = ParallelBackendExecutor(
                    list(names),
                    max_workers=self._max_workers,
                    thread_pool=self._thread_pool,
                    **common_kwargs,
                )
            else:
                executor = BackendExecutor(list(names), **common_kwargs)
            self._executors[key] = executor
        return names, executor

    def summary(self) -> dict[str, Any]:
        reset_calls = sum(
            executor.session_reset_calls for executor in self._executors.values()
        )
        isolation_rejections = sum(
            executor.session_isolation_rejections for executor in self._executors.values()
        )
        parallel_variants = sum(1 for (_names, parallel) in self._executors if parallel)
        sequential_variants = len(self._executors) - parallel_variants
        return {
            "persistent": True,
            "mode": "reuse",
            "isolation_contract": "stateless_or_reset; fresh_only rejected",
            "pool_created": self._thread_pool is not None,
            "max_workers": self._max_workers,
            "execute_calls": self.execute_calls,
            "backend_calls": self.backend_calls,
            "batch_calls": self.batch_calls,
            "executor_variants": len(self._executors),
            "worker_manifest": {
                "backend_order": list(self.backend_names),
                "pool_reused": self._thread_pool is not None,
                "max_workers": self._max_workers,
                "parallel_executor_variants": parallel_variants,
                "sequential_executor_variants": sequential_variants,
                "result_collection_order": "configured_backend_order",
            },
            "reset_calls": reset_calls,
            "isolation_rejections": isolation_rejections,
            "cache": self._cache.summary(),
            "adapter_close_errors": list(self._adapter_close_errors),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
        for backend_name, backend in self.backend_instances.items():
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001
                self._adapter_close_errors.append(
                    {
                        "backend": backend_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

    def __enter__(self) -> "BackendExecutionSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def execute_case(
    case: Case,
    backends: list[str],
    config: ExperimentConfig,
    backend_instances: dict[str, Backend] | None = None,
    *,
    parallel: bool | None = None,
    max_workers: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    use_parallel = (
        bool(getattr(config, "enable_parallel_backend_execution", True))
        if parallel is None
        else bool(parallel)
    )
    executor_cls = ParallelBackendExecutor if use_parallel else BackendExecutor
    executor_kwargs: dict[str, Any] = {}
    if executor_cls is ParallelBackendExecutor:
        executor_kwargs["max_workers"] = max_workers
    executor = executor_cls(
        backends,
        backend_instances=backend_instances,
        backend_factory=make_backend,
        allow_backend_factory_fallback=backend_instances is None,
        **executor_kwargs,
    )
    return executor.execute(case, config)


def build_execution_diagnostic_refs(
    case: Case,
    backends: list[str],
    *,
    case_digest: str,
    raw_results: Mapping[str, Any],
    normalized_results: Mapping[str, Any],
    target_specs: list[Mapping[str, Any]],
    environment: Mapping[str, Any],
    optimizer_config: Mapping[str, Any],
    required_capabilities: tuple[str, ...] | None,
) -> dict[str, Any]:
    """Build non-authoritative opaque refs at the execution metadata boundary."""

    return build_legacy_diagnostic_ref_set(
        backends=backends,
        case_digest=case_digest,
        case_payload=case.to_dict(),
        raw_results=raw_results,
        normalized_results=normalized_results,
        target_specs=target_specs,
        environment=environment,
        optimizer_config=optimizer_config,
        required_capabilities=required_capabilities,
    )
