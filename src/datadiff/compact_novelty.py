"""Small exact bitmaps for bounded seed and semantic-trigger coverage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


COMPACT_NOVELTY_BITMAP_SCHEMA_VERSION = "compact-novelty-bitmap-v1"


class MonotonicSeedTracker:
    """Fast exact tracker for an in-order frozen seed block.

    This is the preferred hot-path structure when a worker consumes a
    contiguous block sequentially.  A bitmap is better reserved for
    out-of-order completion or checkpoint merging.
    """

    __slots__ = ("count", "duplicate_count", "processed", "seed_start")

    def __init__(self, seed_start: int, count: int) -> None:
        self.seed_start = int(seed_start)
        self.count = max(0, int(count))
        self.processed = 0
        self.duplicate_count = 0

    @property
    def expected_seed(self) -> int:
        return self.seed_start + self.processed

    @property
    def complete(self) -> bool:
        return self.processed == self.count

    def observe(self, seed: int) -> bool:
        resolved = int(seed)
        expected = self.expected_seed
        if resolved == expected:
            if self.processed >= self.count:
                raise IndexError("seed tracker is already complete")
            self.processed += 1
            return True
        if self.seed_start <= resolved < expected:
            self.duplicate_count += 1
            return False
        raise ValueError(
            f"out-of-order seed {resolved}; expected contiguous seed {expected}"
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": COMPACT_NOVELTY_BITMAP_SCHEMA_VERSION,
            "kind": "monotonic_seed_cursor",
            "seed_start": self.seed_start,
            "seed_end": self.seed_start + self.count - 1,
            "processed": self.processed,
            "expected_seed": self.expected_seed,
            "duplicate_count": self.duplicate_count,
            "complete": self.complete,
            "state_integer_count": 4,
        }


class DenseBitmap:
    """Exact fixed-size presence bitmap with O(1) observation."""

    __slots__ = ("_count", "_data", "size")

    def __init__(self, size: int) -> None:
        normalized = max(0, int(size))
        self.size = normalized
        self._data = bytearray((normalized + 7) // 8)
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    @property
    def byte_size(self) -> int:
        return len(self._data)

    def contains(self, index: int) -> bool:
        resolved = self._validate_index(index)
        return bool(self._data[resolved >> 3] & (1 << (resolved & 7)))

    def observe(self, index: int) -> bool:
        """Mark an index and return ``True`` only on its first observation."""

        resolved = self._validate_index(index)
        byte_index = resolved >> 3
        mask = 1 << (resolved & 7)
        if self._data[byte_index] & mask:
            return False
        self._data[byte_index] |= mask
        self._count += 1
        return True

    def snapshot(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": COMPACT_NOVELTY_BITMAP_SCHEMA_VERSION,
            "encoding": "one-bit-little-endian",
            "size": self.size,
            "observed_count": self.count,
            "byte_size": self.byte_size,
            "density": self.count / self.size if self.size else 0.0,
        }
        if include_data:
            payload["data"] = self._data.hex()
        return payload

    def _validate_index(self, index: int) -> int:
        resolved = int(index)
        if resolved < 0 or resolved >= self.size:
            raise IndexError(f"bitmap index {resolved} outside [0, {self.size})")
        return resolved


class SeedRangeBitmap:
    """Exact processed-seed tracking for one frozen contiguous range."""

    __slots__ = ("bitmap", "count", "seed_start")

    def __init__(self, seed_start: int, count: int) -> None:
        self.seed_start = int(seed_start)
        self.count = max(0, int(count))
        self.bitmap = DenseBitmap(self.count)

    def contains(self, seed: int) -> bool:
        return self.bitmap.contains(self._index(seed))

    def observe(self, seed: int) -> bool:
        return self.bitmap.observe(self._index(seed))

    def snapshot(self, *, include_data: bool = False) -> dict[str, Any]:
        return {
            **self.bitmap.snapshot(include_data=include_data),
            "kind": "frozen_contiguous_seed_range",
            "seed_start": self.seed_start,
            "seed_end": self.seed_start + self.count - 1,
        }

    def _index(self, seed: int) -> int:
        return int(seed) - self.seed_start


class DenseProductBitmap:
    """Exact bitmap over a small Cartesian product of categorical dimensions."""

    __slots__ = ("_lookups", "_strides", "bitmap", "dimensions")

    def __init__(self, dimensions: Sequence[Sequence[str]]) -> None:
        normalized = tuple(tuple(str(value) for value in values) for values in dimensions)
        if not normalized or any(not values for values in normalized):
            raise ValueError("dense product dimensions must be non-empty")
        if any(len(set(values)) != len(values) for values in normalized):
            raise ValueError("dense product dimension values must be unique")
        self.dimensions = normalized
        self._lookups = tuple(
            {value: index for index, value in enumerate(values)}
            for values in normalized
        )
        strides = []
        stride = 1
        for values in reversed(normalized):
            strides.append(stride)
            stride *= len(values)
        self._strides = tuple(reversed(strides))
        self.bitmap = DenseBitmap(stride)

    def encode(self, values: Sequence[str]) -> int:
        resolved = tuple(str(value) for value in values)
        if len(resolved) != len(self.dimensions):
            raise ValueError(
                f"expected {len(self.dimensions)} dimensions, got {len(resolved)}"
            )
        index = 0
        for position, value in enumerate(resolved):
            try:
                value_index = self._lookups[position][value]
            except KeyError as exc:
                raise KeyError(
                    f"unknown value {value!r} for dense dimension {position}"
                ) from exc
            index += value_index * self._strides[position]
        return index

    def contains(self, values: Sequence[str]) -> bool:
        return self.bitmap.contains(self.encode(values))

    def observe(self, values: Sequence[str]) -> tuple[int, bool]:
        index = self.encode(values)
        return index, self.bitmap.observe(index)

    def snapshot(self, *, include_data: bool = False) -> dict[str, Any]:
        return {
            **self.bitmap.snapshot(include_data=include_data),
            "kind": "dense_categorical_product",
            "dimension_sizes": [len(values) for values in self.dimensions],
            "dimension_count": len(self.dimensions),
        }
