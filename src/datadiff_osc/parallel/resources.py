"""Deterministic accounting for frozen ``ResourceTokens`` requests."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from datadiff_osc.schemas import ResourceTokens


class ResourceRequestTooLarge(ValueError):
    """A task can never fit the declared runtime capacity."""


@dataclass(frozen=True, slots=True)
class ResourceCapacity:
    cpu_tokens: int
    rss_bytes: int
    io_slots: tuple[tuple[str, int], ...]
    backend_internal_threads: int

    def __post_init__(self) -> None:
        if self.cpu_tokens < 1 or self.rss_bytes < 0 or self.backend_internal_threads < 0:
            raise ValueError("resource capacity is outside its valid range")
        names = tuple(name for name, _ in self.io_slots)
        if not names or len(names) != len(set(names)):
            raise ValueError("I/O capacity classes must be non-empty and unique")
        if any(not name or slots < 1 for name, slots in self.io_slots):
            raise ValueError("I/O capacity entries require a name and positive slots")

    @classmethod
    def build(
        cls,
        *,
        cpu_tokens: int,
        rss_bytes: int,
        io_slots: dict[str, int],
        backend_internal_threads: int,
    ) -> "ResourceCapacity":
        return cls(
            cpu_tokens=int(cpu_tokens),
            rss_bytes=int(rss_bytes),
            io_slots=tuple(sorted((str(name), int(value)) for name, value in io_slots.items())),
            backend_internal_threads=int(backend_internal_threads),
        )

    @property
    def io_limits(self) -> dict[str, int]:
        return dict(self.io_slots)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    cpu_tokens: int
    rss_bytes: int
    io_in_use: tuple[tuple[str, int], ...]
    backend_internal_threads: int
    exclusive_states: tuple[str, ...]


class ResourcePool:
    """Atomic acquire/release for every frozen resource-token dimension."""

    def __init__(self, capacity: ResourceCapacity) -> None:
        self.capacity = capacity
        self._cpu = 0
        self._rss = 0
        self._io = {name: 0 for name, _ in capacity.io_slots}
        self._internal_threads = 0
        self._exclusive: set[str] = set()
        self._lock = RLock()

    def validate_request(self, request: ResourceTokens) -> None:
        limits = self.capacity.io_limits
        reasons: list[str] = []
        if request.cpu_tokens > self.capacity.cpu_tokens:
            reasons.append("cpu_tokens")
        if request.rss_bytes > self.capacity.rss_bytes:
            reasons.append("rss_bytes")
        if request.backend_internal_threads > self.capacity.backend_internal_threads:
            reasons.append("backend_internal_threads")
        if request.io_class not in limits:
            reasons.append("io_class")
        if reasons:
            raise ResourceRequestTooLarge(
                "resource request exceeds capacity: " + ",".join(reasons)
            )

    def try_acquire(self, request: ResourceTokens) -> bool:
        self.validate_request(request)
        limits = self.capacity.io_limits
        with self._lock:
            if self._cpu + request.cpu_tokens > self.capacity.cpu_tokens:
                return False
            if self._rss + request.rss_bytes > self.capacity.rss_bytes:
                return False
            if (
                self._internal_threads + request.backend_internal_threads
                > self.capacity.backend_internal_threads
            ):
                return False
            if self._io[request.io_class] + 1 > limits[request.io_class]:
                return False
            if request.exclusive_state and request.exclusive_state in self._exclusive:
                return False
            self._cpu += request.cpu_tokens
            self._rss += request.rss_bytes
            self._internal_threads += request.backend_internal_threads
            self._io[request.io_class] += 1
            if request.exclusive_state:
                self._exclusive.add(request.exclusive_state)
            return True

    def release(self, request: ResourceTokens) -> None:
        with self._lock:
            if (
                self._cpu < request.cpu_tokens
                or self._rss < request.rss_bytes
                or self._internal_threads < request.backend_internal_threads
                or self._io.get(request.io_class, 0) < 1
                or (request.exclusive_state and request.exclusive_state not in self._exclusive)
            ):
                raise RuntimeError("resource release does not match an active reservation")
            self._cpu -= request.cpu_tokens
            self._rss -= request.rss_bytes
            self._internal_threads -= request.backend_internal_threads
            self._io[request.io_class] -= 1
            if request.exclusive_state:
                self._exclusive.remove(request.exclusive_state)

    @property
    def snapshot(self) -> ResourceSnapshot:
        with self._lock:
            return ResourceSnapshot(
                cpu_tokens=self._cpu,
                rss_bytes=self._rss,
                io_in_use=tuple(sorted(self._io.items())),
                backend_internal_threads=self._internal_threads,
                exclusive_states=tuple(sorted(self._exclusive)),
            )

    @property
    def is_idle(self) -> bool:
        snapshot = self.snapshot
        return (
            snapshot.cpu_tokens == 0
            and snapshot.rss_bytes == 0
            and snapshot.backend_internal_threads == 0
            and not snapshot.exclusive_states
            and all(value == 0 for _, value in snapshot.io_in_use)
        )
