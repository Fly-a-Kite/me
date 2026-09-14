"""Tests for the datafusion_parallel target (multi-partition execution surface)."""

from __future__ import annotations

from datadiff.targets import instantiate_target_backend, target_spec


def test_datafusion_parallel_is_registered() -> None:
    spec = target_spec("datafusion_parallel")
    assert spec.family == "query_engine"
    assert spec.backend == "datafusion_parallel"


def test_datafusion_parallel_uses_multiple_partitions() -> None:
    backend = instantiate_target_backend("datafusion_parallel")
    assert backend.name == "datafusion_parallel"
    assert backend._target_partitions >= 2


def test_datafusion_parallel_partitions_are_configurable(monkeypatch) -> None:
    monkeypatch.setenv("DATADIFF_DATAFUSION_PARALLEL_PARTITIONS", "4")
    backend = instantiate_target_backend("datafusion_parallel")
    assert backend._target_partitions == 4
