"""Hash-seed independent helpers for generation and campaign replay.

The random generator may only consume ordered sequences.  This module gives
generation and mutation code one canonical conversion point instead of relying
on the iteration order of a ``set`` or ``dict``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from datadiff.canonicalization import canonical_key, short_canonical_hash


_T = TypeVar("_T")
RANDOMNESS_SCHEMA_VERSION = "datadiff-randomness-v1"


def canonical_candidates(
    values: Iterable[_T] | Mapping[Any, _T],
    *,
    key: Callable[[_T], Any] | None = None,
) -> list[_T]:
    """Return candidates in a process- and hash-seed-independent order.

    ``sorted`` alone is insufficient for mixed, structured values.  Sorting by
    a canonical digest makes the rule work for strings, tuples, dataclasses and
    mappings while preserving duplicate values (which can encode a weight).
    """

    iterable: Iterable[_T]
    if isinstance(values, Mapping):
        iterable = values.values()
    else:
        iterable = values
    materialized = list(iterable)
    value_key = key or (lambda value: value)
    return sorted(
        materialized,
        key=lambda value: (
            canonical_key(_canonical_payload(value_key(value))),
            repr(type(value)),
        ),
    )


def derive_case_seed(root_seed: int, case_index: int, source: str) -> int:
    """Derive a worker-independent 64-bit case seed from assigned identity."""

    payload = f"{int(root_seed)}:{int(case_index)}:{str(source)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def program_digest(program_payload: Any) -> str:
    return f"program-{short_canonical_hash(_canonical_payload(program_payload), 64)}"


def case_digest(case_payload: Any) -> str:
    return f"case-{short_canonical_hash(_canonical_payload(case_payload), 64)}"


@dataclass(frozen=True, slots=True)
class CaseSeedAssignment:
    root_seed: int
    case_index: int
    source: str
    derived_seed: int
    generator_version: str
    mutator_version: str
    schema_version: str = RANDOMNESS_SCHEMA_VERSION

    @classmethod
    def allocate(
        cls,
        *,
        root_seed: int,
        case_index: int,
        source: str,
        generator_version: str,
        mutator_version: str = "",
    ) -> "CaseSeedAssignment":
        return cls(
            root_seed=int(root_seed),
            case_index=int(case_index),
            source=str(source),
            derived_seed=derive_case_seed(root_seed, case_index, source),
            generator_version=str(generator_version),
            mutator_version=str(mutator_version),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_payload(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical_payload(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, list):
        return [_canonical_payload(item) for item in value]
    if isinstance(value, set | frozenset):
        return canonical_candidates((_canonical_payload(item) for item in value))
    return value
