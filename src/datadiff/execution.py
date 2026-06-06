from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from datadiff.backends import make_backend
from datadiff.backends.base import Backend, prepare_tables
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.normalizer import normalize_result

BackendFactory = Callable[[str], Backend]


class BackendExecutor:
    def __init__(
        self,
        backend_names: list[str],
        *,
        backend_instances: dict[str, Backend] | None = None,
        backend_factory: BackendFactory = make_backend,
        allow_backend_factory_fallback: bool = True,
    ) -> None:
        self.backend_names = list(backend_names)
        self._backend_instances = dict(backend_instances or {})
        self._backend_factory = backend_factory
        self._allow_backend_factory_fallback = allow_backend_factory_fallback

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
        prepared_tables = prepare_tables(case.tables)
        for backend_name in self.backend_names:
            raw_row, normalized_row = self._execute_one(
                backend_name,
                prepared_tables,
                case,
                config,
            )
            raw_results[backend_name] = raw_row
            normalized[backend_name] = normalized_row
        return raw_results, normalized

    def _execute_one(
        self,
        backend_name: str,
        prepared_tables: list[Any],
        case: Case,
        config: ExperimentConfig,
    ) -> tuple[dict[str, Any], Any]:
        backend = self._backend_for_name(backend_name)
        result = backend.run(prepared_tables, case.program)
        raw_row = {
            key: value
            for key, value in result.to_dict().items()
            if key != "data"
        }
        normalized_row = normalize_result(
            result,
            case.program,
            enable_normalizer=config.enable_normalizer,
        )
        return raw_row, normalized_row

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
    ) -> None:
        super().__init__(
            backend_names,
            backend_instances=backend_instances,
            backend_factory=backend_factory,
            allow_backend_factory_fallback=allow_backend_factory_fallback,
        )
        self._max_workers = max_workers

    def execute(
        self,
        case: Case,
        config: ExperimentConfig,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        if len(self.backend_names) <= 1 or len(set(self.backend_names)) != len(self.backend_names):
            return super().execute(case, config)

        for backend_name in self.backend_names:
            self._backend_for_name(backend_name)

        prepared_tables = prepare_tables(case.tables)
        worker_count = max(1, min(len(self.backend_names), int(self._max_workers or len(self.backend_names))))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                backend_name: pool.submit(
                    self._execute_one,
                    backend_name,
                    prepared_tables,
                    case,
                    config,
                )
                for backend_name in self.backend_names
            }
            rows = {
                backend_name: future.result()
                for backend_name, future in futures.items()
            }

        raw_results: dict[str, dict[str, Any]] = {}
        normalized: dict[str, Any] = {}
        for backend_name in self.backend_names:
            raw_row, normalized_row = rows[backend_name]
            raw_results[backend_name] = raw_row
            normalized[backend_name] = normalized_row
        return raw_results, normalized


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
