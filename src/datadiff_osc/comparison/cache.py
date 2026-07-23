"""Evidence-tier-safe bounded result caching.

The frozen :class:`RuntimeCacheKey` remains the storage identity.  Both cache
boundaries independently bind that key to the endpoint, staged request, and
stored value, and reject uncached evidence tiers.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, TypeVar

from datadiff_osc._canonical import assert_deeply_immutable, stable_digest
from datadiff_osc.contract_engine.model import Endpoint
from datadiff_osc.contract_engine.planner import ComparisonStage
from datadiff_osc.schemas import (
    ContractFingerprint,
    EvidenceTier,
    RuntimeCacheKey,
    STAGED_COMPARISON_SCHEMA_VERSION,
    StagedComparisonRequest,
    StagedComparisonResult,
)


T = TypeVar("T")


class CachePolicyError(ValueError):
    """Raised when an uncached evidence tier reaches a cache boundary."""


class CacheBindingError(ValueError):
    """A cache key, request, endpoint, or value is not mutually bound."""


@dataclass(frozen=True, slots=True)
class CacheValueBinding:
    key_digest: str
    endpoint_digest: str
    endpoint_id: str
    contract_fingerprint: ContractFingerprint
    semantic_schema_version: str
    evidence_tier: EvidenceTier
    request_digest: str
    value_digest: str
    schema_version: str = "osc-runtime-cache-value-binding-v2"

    def __post_init__(self) -> None:
        for value in (
            self.key_digest,
            self.endpoint_digest,
            self.endpoint_id,
            self.semantic_schema_version,
            self.request_digest,
            self.value_digest,
        ):
            if not value:
                raise ValueError("cache value binding fields must be non-empty")


@dataclass(frozen=True, slots=True)
class _CacheEntry(Generic[T]):
    key: RuntimeCacheKey
    binding: CacheValueBinding
    value: T
    size_bytes: int


def build_runtime_cache_key(
    endpoint: Endpoint,
    contract_fingerprint: ContractFingerprint,
    *,
    semantic_schema_version: str,
    evidence_tier: EvidenceTier,
) -> RuntimeCacheKey:
    """Build the frozen complete runtime cache identity for an endpoint."""

    return RuntimeCacheKey(
        endpoint_digest=endpoint.digest,
        contract_fingerprint=contract_fingerprint,
        semantic_schema_version=semantic_schema_version,
        backend=endpoint.backend,
        backend_version=endpoint.backend_version,
        adapter_revision=endpoint.adapter_revision,
        execution_mode=endpoint.execution_mode,
        physical_layout=endpoint.physical_layout,
        evidence_tier=evidence_tier,
    )


class RuntimeResultCache(Generic[T]):
    """Thread-safe bounded LRU keyed only by ``RuntimeCacheKey.digest``."""

    def __init__(self, *, max_entries: int = 4096, max_bytes: int = 64 << 20) -> None:
        if max_entries < 0 or max_bytes < 0:
            raise ValueError("cache limits must be non-negative")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[str, _CacheEntry[T]] = OrderedDict()
        self._bytes = 0
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.rejected_oversize = 0

    @staticmethod
    def _require_cacheable(key: RuntimeCacheKey) -> None:
        if not key.cache_allowed:
            raise CachePolicyError(
                f"evidence tier {key.evidence_tier.value} is never cacheable"
            )

    @property
    def current_entries(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._bytes

    @staticmethod
    def _request_binding_errors(
        key: RuntimeCacheKey,
        endpoint: Endpoint,
        request: StagedComparisonRequest,
    ) -> tuple[str, ...]:
        errors: list[str] = []
        expected_key = build_runtime_cache_key(
            endpoint,
            request.contract_fingerprint,
            semantic_schema_version=key.semantic_schema_version,
            evidence_tier=request.evidence_tier,
        )
        if expected_key != key:
            errors.append("cache_key_endpoint_contract_schema_or_tier_mismatch")
        if request.schema_version != STAGED_COMPARISON_SCHEMA_VERSION:
            errors.append("staged_request_schema_version_mismatch")
        if endpoint.endpoint_id not in request.endpoint_ids:
            errors.append("cache_endpoint_not_bound_to_staged_request")
        return tuple(errors)

    @staticmethod
    def _value_binding_errors(
        key: RuntimeCacheKey,
        endpoint: Endpoint,
        request: StagedComparisonRequest,
        value: T,
    ) -> tuple[str, ...]:
        errors = list(
            RuntimeResultCache._request_binding_errors(key, endpoint, request)
        )
        if isinstance(value, StagedComparisonResult):
            if value.request_digest != request.digest:
                errors.append("cached_result_request_digest_mismatch")
            if (
                value.evidence_tier is not request.evidence_tier
                or value.evidence_tier is not key.evidence_tier
            ):
                errors.append("cached_result_evidence_tier_mismatch")
            if value.schema_version != request.schema_version:
                errors.append("cached_result_schema_version_mismatch")
            if value.endpoint_order != request.endpoint_ids:
                errors.append("cached_result_endpoint_order_mismatch")

            exact_stage = ComparisonStage.S3_EXACT_MATERIALIZED.value
            cacheable_stages = {
                ComparisonStage.S0_STATIC.value,
                ComparisonStage.S1_STATUS_SCHEMA_CARDINALITY.value,
                ComparisonStage.S2_COMPONENT_FINGERPRINT.value,
                exact_stage,
            }
            if value.comparison_stage not in cacheable_stages:
                errors.append("cached_result_comparison_stage_not_cacheable")
            if value.exact_escalated != (value.comparison_stage == exact_stage):
                errors.append("cached_result_exact_stage_flag_mismatch")
            if request.exact_required and (
                not value.exact_escalated
                or value.comparison_stage != exact_stage
            ):
                errors.append("exact_required_cached_result_is_not_exact_materialized")
        elif request.exact_required:
            errors.append(
                "exact_required_cache_value_lacks_exact_result_binding"
            )
        return tuple(errors)

    @staticmethod
    def _binding(
        key: RuntimeCacheKey,
        endpoint: Endpoint,
        request: StagedComparisonRequest,
        value: T,
    ) -> CacheValueBinding:
        errors = RuntimeResultCache._value_binding_errors(
            key, endpoint, request, value
        )
        if errors:
            raise CacheBindingError(";".join(errors))
        return CacheValueBinding(
            key_digest=key.digest,
            endpoint_digest=endpoint.digest,
            endpoint_id=endpoint.endpoint_id,
            contract_fingerprint=request.contract_fingerprint,
            semantic_schema_version=key.semantic_schema_version,
            evidence_tier=key.evidence_tier,
            request_digest=request.digest,
            value_digest=stable_digest("osc-runtime-cache-value", value),
        )

    def get(
        self,
        key: RuntimeCacheKey,
        *,
        endpoint: Endpoint,
        request: StagedComparisonRequest,
    ) -> T | None:
        self._require_cacheable(key)
        request_errors = self._request_binding_errors(key, endpoint, request)
        if request_errors:
            raise CacheBindingError(";".join(request_errors))
        with self._lock:
            entry = self._entries.get(key.digest)
            if entry is None:
                self.misses += 1
                return None
            if entry.key != key:
                raise RuntimeError("cache digest collision across distinct frozen keys")
            expected_binding = self._binding(key, endpoint, request, entry.value)
            if entry.binding != expected_binding:
                raise CacheBindingError("stored cache value binding is invalid")
            self._entries.move_to_end(key.digest)
            self.hits += 1
            return entry.value

    def put(
        self,
        key: RuntimeCacheKey,
        value: T,
        *,
        endpoint: Endpoint,
        request: StagedComparisonRequest,
        size_bytes: int,
    ) -> bool:
        self._require_cacheable(key)
        assert_deeply_immutable(value)
        binding = self._binding(key, endpoint, request, value)
        if size_bytes < 0:
            raise ValueError("cache entry size must be non-negative")
        if self.max_entries == 0 or self.max_bytes == 0 or size_bytes > self.max_bytes:
            with self._lock:
                self.rejected_oversize += 1
            return False
        entry = _CacheEntry(
            key=key,
            binding=binding,
            value=value,
            size_bytes=int(size_bytes),
        )
        with self._lock:
            prior = self._entries.pop(key.digest, None)
            if prior is not None:
                if prior.key != key:
                    raise RuntimeError("cache digest collision across distinct frozen keys")
                self._bytes -= prior.size_bytes
            self._entries[key.digest] = entry
            self._bytes += entry.size_bytes
            while (
                len(self._entries) > self.max_entries
                or self._bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._bytes -= evicted.size_bytes
                self.evictions += 1
        return True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0
