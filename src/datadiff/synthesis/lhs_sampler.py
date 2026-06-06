from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SchemaSpec:
    column_count: int
    row_count: int
    null_density: float
    type_mix: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_count": self.column_count,
            "row_count": self.row_count,
            "null_density": self.null_density,
            "type_mix": list(self.type_mix),
        }


def lhs_schemas(
    n_samples: int,
    *,
    column_count_range: tuple[int, int] = (2, 8),
    row_count_range: tuple[int, int] = (5, 200),
    null_density_range: tuple[float, float] = (0.0, 0.4),
    type_universe: tuple[str, ...] = ("numeric", "string", "bool", "datetime", "decimal"),
    rnd: random.Random,
) -> list[SchemaSpec]:
    count = max(0, int(n_samples))
    if count <= 0:
        return []
    column_bins = _permuted_bins(count, rnd)
    row_bins = _permuted_bins(count, rnd)
    null_bins = _permuted_bins(count, rnd)
    type_bins = _permuted_bins(count, rnd)
    specs: list[SchemaSpec] = []
    for index in range(count):
        column_count = _int_from_bin(
            column_bins[index],
            count,
            low=column_count_range[0],
            high=column_count_range[1],
        )
        row_count = _int_from_bin(
            row_bins[index],
            count,
            low=row_count_range[0],
            high=row_count_range[1],
        )
        null_density = _float_from_bin(
            null_bins[index],
            count,
            low=null_density_range[0],
            high=null_density_range[1],
        )
        type_mix = _type_mix_from_bin(
            type_bins[index],
            count,
            column_count=column_count,
            type_universe=type_universe,
            rnd=rnd,
        )
        specs.append(
            SchemaSpec(
                column_count=max(1, column_count),
                row_count=max(0, row_count),
                null_density=max(0.0, min(1.0, null_density)),
                type_mix=type_mix,
            )
        )
    return specs


def schema_spec_for_seed(seed: int, *, sample_count: int = 256) -> SchemaSpec:
    count = max(1, int(sample_count))
    rnd = random.Random(0x5EED_1A5 ^ int(seed // count))
    return lhs_schemas(count, rnd=rnd)[int(seed) % count]


def _permuted_bins(count: int, rnd: random.Random) -> list[int]:
    bins = list(range(count))
    rnd.shuffle(bins)
    return bins


def _int_from_bin(bin_index: int, bin_count: int, *, low: int, high: int) -> int:
    lo = min(int(low), int(high))
    hi = max(int(low), int(high))
    if bin_count <= 1:
        return lo
    value = lo + ((hi - lo) * int(bin_index)) / max(1, bin_count - 1)
    return int(round(value))


def _float_from_bin(bin_index: int, bin_count: int, *, low: float, high: float) -> float:
    lo = min(float(low), float(high))
    hi = max(float(low), float(high))
    if bin_count <= 0:
        return lo
    jitter = 0.5 / max(1, bin_count)
    center = (float(bin_index) + 0.5) / max(1, bin_count)
    return lo + (hi - lo) * min(1.0, max(0.0, center + jitter))


def _type_mix_from_bin(
    bin_index: int,
    bin_count: int,
    *,
    column_count: int,
    type_universe: tuple[str, ...],
    rnd: random.Random,
) -> tuple[str, ...]:
    universe = tuple(type_universe) or ("numeric",)
    width = max(1, min(len(universe), 1 + int((bin_index / max(1, bin_count)) * len(universe))))
    start = bin_index % len(universe)
    selected = [universe[(start + offset) % len(universe)] for offset in range(width)]
    while len(selected) < max(1, column_count):
        selected.append(rnd.choice(selected or list(universe)))
    rnd.shuffle(selected)
    return tuple(selected[: max(1, column_count)])
