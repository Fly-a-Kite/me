from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any

from datadiff.backends.base import Backend
from datadiff.canonicalization import short_canonical_hash
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.execution_lattice import ExecutionCacheKey, ExecutionNode
from datadiff.ir_runtime import resolve_case_program
from datadiff.normalizer import NormalizedResult
from datadiff.targets import target_spec


CachedExecutionRow = tuple[dict[str, Any], NormalizedResult]


@dataclass(frozen=True, slots=True)
class CachedExecutionEntry:
    row: CachedExecutionRow
    size_bytes: int
    provenance: dict[str, Any]


class ExecutionResultCache:
    """Bounded LRU store keyed by case, execution node, environment, and adapter semantics."""

    def __init__(
        self,
        backend_instances: Mapping[str, Backend],
        *,
        environment: Mapping[str, Any] | None = None,
        adapter_revision: str = "",
        limit: int = 4096,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self._backend_instances = backend_instances
        self._environment = dict(environment or {})
        self._environment_digest = short_canonical_hash(self._environment, 64)
        self._adapter_revision = str(adapter_revision or "")
        self._limit = max(0, int(limit or 0))
        self._max_bytes = max(0, int(max_bytes or 0))
        self._rows: OrderedDict[str, CachedExecutionEntry] = OrderedDict()
        self._nodes: dict[tuple[str, str, str, str], ExecutionNode] = {}
        self._key_provenance: dict[str, dict[str, Any]] = {}
        self._lock = RLock()
        self.current_bytes = 0
        self.evictions = 0
        self.rejected_oversize = 0
        self.hits = 0
        self.misses = 0

    def enabled_for(self, config: ExperimentConfig) -> bool:
        return bool(
            self._limit > 0
            and self._max_bytes > 0
            and config.method_policy.execution.cache_mode == "lattice"
        )

    @staticmethod
    def case_digest(case: Case, config: ExperimentConfig) -> str:
        payload = case.to_dict()
        resolved = resolve_case_program(case, config)
        return short_canonical_hash(
            {
                "tables": payload.get("tables", []),
                "ir_mode": resolved.ir_mode,
                "program_ir_digest": resolved.program_ir_digest,
            },
            64,
        )

    def key(
        self,
        *,
        case_digest: str,
        backend_name: str,
        config: ExperimentConfig,
        input_physical_layout: str | None = None,
        physical_plan_detail: str | None = None,
    ) -> str:
        backend = self._backend_instances[backend_name]
        resolved_layout = str(
            backend.input_physical_layout
            if input_physical_layout is None
            else input_physical_layout
        )
        node = self._execution_node(
            backend_name,
            input_physical_layout=resolved_layout,
        )
        execution_semantics = {
            "adapter_revision": self._adapter_revision,
            "backend_class": (
                f"{type(backend).__module__}."
                f"{type(backend).__qualname__}"
            ),
            "adapter_spi": backend.spi_manifest(),
            "input_physical_layout": resolved_layout,
            "enable_normalizer": bool(config.enable_normalizer),
            "target_version": str(config.target_version or ""),
            "fixed_version": str(config.fixed_version or ""),
            "method_policy": config.method_policy.to_dict(),
            "evidence_tier": config.evidence_tier,
            "physical_plan_detail": (
                config.method_policy.execution.plan_collection.detail_for(
                    config.evidence_tier
                )
                if physical_plan_detail is None
                else str(physical_plan_detail)
            ),
        }
        execution_semantics_digest = short_canonical_hash(execution_semantics, 32)
        cache_key = ExecutionCacheKey(
            case_digest=case_digest,
            node_id=node.node_id,
            environment_digest=self._environment_digest,
            adapter_revision=execution_semantics_digest,
        ).key
        with self._lock:
            self._key_provenance[cache_key] = {
                "case_digest": case_digest,
                "node": node.to_dict(),
                "environment_digest": self._environment_digest,
                "execution_semantics_digest": execution_semantics_digest,
                "execution_semantics": execution_semantics,
            }
        return cache_key

    def get(self, cache_key: str) -> CachedExecutionRow | None:
        with self._lock:
            entry = self._rows.pop(cache_key, None)
            if entry is None:
                self.misses += 1
                return None
            self._rows[cache_key] = entry
            self.hits += 1
        raw, normalized = self._clone(entry.row)
        raw["execution_cache_hit"] = True
        raw["execution_cache_lookup"] = "hit"
        raw["execution_cache_key"] = cache_key
        raw["execution_cache_entry_bytes"] = entry.size_bytes
        raw["execution_cache_provenance"] = dict(entry.provenance)
        return raw, normalized

    def put(self, cache_key: str, row: CachedExecutionRow) -> None:
        if self._limit <= 0 or self._max_bytes <= 0:
            return
        raw, _normalized = row
        with self._lock:
            provenance = dict(self._key_provenance.get(cache_key, {}))
        raw["execution_cache_hit"] = False
        raw["execution_cache_lookup"] = "miss"
        raw["execution_cache_key"] = cache_key
        raw["execution_cache_provenance"] = provenance
        raw["execution_cache_store"] = "pending"
        size_bytes = 0
        for _attempt in range(3):
            size_bytes = self._row_size_bytes(row)
            raw["execution_cache_entry_bytes"] = size_bytes
        if size_bytes > self._max_bytes:
            raw["execution_cache_store"] = "rejected_oversize"
            with self._lock:
                self.rejected_oversize += 1
            return
        raw["execution_cache_store"] = "stored"
        entry = CachedExecutionEntry(
            row=self._clone(row),
            size_bytes=size_bytes,
            provenance=provenance,
        )
        with self._lock:
            previous = self._rows.pop(cache_key, None)
            if previous is not None:
                self.current_bytes -= previous.size_bytes
            self._rows[cache_key] = entry
            self.current_bytes += entry.size_bytes
            while (
                len(self._rows) > self._limit
                or self.current_bytes > self._max_bytes
            ):
                evicted_key, evicted = self._rows.popitem(last=False)
                self.current_bytes -= evicted.size_bytes
                self._key_provenance.pop(evicted_key, None)
                self.evictions += 1

    def summary(self) -> dict[str, Any]:
        observations = self.hits + self.misses
        with self._lock:
            return {
                "enabled_for_lattice_arms": self._limit > 0 and self._max_bytes > 0,
                "limit": self._limit,
                "max_bytes": self._max_bytes,
                "entries": len(self._rows),
                "current_bytes": self.current_bytes,
                "evictions": self.evictions,
                "rejected_oversize": self.rejected_oversize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / observations if observations else 0.0,
                "environment_digest": self._environment_digest,
                "adapter_revision": self._adapter_revision,
                "concurrency_safe": True,
            }

    def _execution_node(
        self,
        backend_name: str,
        *,
        input_physical_layout: str | None = None,
    ) -> ExecutionNode:
        backend = self._backend_instances[backend_name]
        resolved_layout = str(
            backend.input_physical_layout
            if input_physical_layout is None
            else input_physical_layout
        )
        try:
            spec = target_spec(backend_name)
            family = spec.family
            api_lowering = spec.execution_model
            adapter = spec.adapter
            execution_mode = spec.capability_model.execution_modes[0]
        except ValueError:
            family = "custom"
            api_lowering = "custom"
            adapter = f"custom:{type(backend).__module__}.{type(backend).__qualname__}"
            execution_mode = _execution_mode(backend_name, family)
        version_key = backend_name.split("_", 1)[0]
        version = str(
            self._environment.get(
                backend_name,
                self._environment.get(version_key, ""),
            )
            or ""
        )
        node_key = (
            backend_name,
            execution_mode,
            resolved_layout,
            version,
        )
        node = self._nodes.get(node_key)
        if node is not None:
            return node
        node = ExecutionNode.build(
            backend=backend_name,
            family=family,
            api_lowering=api_lowering,
            execution_mode=execution_mode,
            input_layout=resolved_layout,
            optimizer_config={},
            version=version,
            adapter=adapter,
        )
        self._nodes[node_key] = node
        return node

    @staticmethod
    def _clone(row: CachedExecutionRow) -> CachedExecutionRow:
        raw, normalized = row
        return dict(raw), NormalizedResult.from_dict(normalized.to_dict())

    @staticmethod
    def _row_size_bytes(row: CachedExecutionRow) -> int:
        raw, normalized = row
        return len(
            json.dumps(
                {"raw": raw, "normalized": normalized.to_dict()},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )


def _execution_mode(backend_name: str, family: str) -> str:
    name = backend_name.lower()
    if "streaming" in name:
        return "streaming"
    if "lazy" in name:
        return "lazy"
    if name == "polars" or family == "dataframe":
        return "eager"
    if family == "arrow":
        return "batch"
    return "default"
