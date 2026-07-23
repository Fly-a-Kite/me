from __future__ import annotations

import pytest

from datadiff_osc.parallel.resources import (
    ResourceCapacity,
    ResourcePool,
    ResourceRequestTooLarge,
)
from datadiff_osc.schemas import ResourceTokens


def _capacity():
    return ResourceCapacity.build(
        cpu_tokens=4,
        rss_bytes=100,
        io_slots={"default": 2, "storage": 1},
        backend_internal_threads=4,
    )


def _request(**changes):
    values = {
        "cpu_tokens": 1,
        "rss_bytes": 20,
        "io_class": "default",
        "backend_internal_threads": 1,
        "exclusive_state": "",
    }
    values.update(changes)
    return ResourceTokens(**values)


def test_pool_accounts_every_resource_dimension_and_releases_cleanly():
    pool = ResourcePool(_capacity())
    first = _request(exclusive_state="catalog-a")
    second = _request(exclusive_state="catalog-a")
    assert pool.try_acquire(first)
    assert pool.try_acquire(second) is False
    snapshot = pool.snapshot
    assert snapshot.cpu_tokens == 1
    assert snapshot.rss_bytes == 20
    assert snapshot.backend_internal_threads == 1
    assert snapshot.exclusive_states == ("catalog-a",)
    pool.release(first)
    assert pool.is_idle


def test_io_slots_cpu_rss_and_internal_threads_backpressure():
    pool = ResourcePool(_capacity())
    storage = _request(io_class="storage")
    assert pool.try_acquire(storage)
    assert pool.try_acquire(storage) is False
    pool.release(storage)
    heavy = _request(cpu_tokens=4, rss_bytes=100, backend_internal_threads=4)
    assert pool.try_acquire(heavy)
    assert pool.try_acquire(_request()) is False
    pool.release(heavy)


@pytest.mark.parametrize(
    "tokens",
    [
        _request(cpu_tokens=5),
        _request(rss_bytes=101),
        _request(backend_internal_threads=5),
        _request(io_class="unknown"),
    ],
)
def test_impossible_request_fails_before_execution(tokens):
    with pytest.raises(ResourceRequestTooLarge, match="exceeds capacity"):
        ResourcePool(_capacity()).validate_request(tokens)
