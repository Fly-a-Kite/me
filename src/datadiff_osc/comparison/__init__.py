"""Exact-preserving comparison orchestration for OSC runtime consumers."""

from datadiff_osc.comparison.cache import (
    CachePolicyError,
    RuntimeResultCache,
    build_runtime_cache_key,
)
from datadiff_osc.comparison.staged import (
    FingerprintCluster,
    FingerprintClustering,
    StagedComparisonEngine,
    cluster_component_fingerprints,
)

__all__ = [
    "CachePolicyError",
    "FingerprintCluster",
    "FingerprintClustering",
    "RuntimeResultCache",
    "StagedComparisonEngine",
    "build_runtime_cache_key",
    "cluster_component_fingerprints",
]
