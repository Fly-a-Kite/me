"""Private fail-closed helpers for Phase-6 typed semantic replay.

The canonical transport deliberately has no object registry.  Domain-owned
replayers use the small validators in this module to reconstruct their own
types without weakening the frozen canonical encoding or creating a second
authority surface here.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable
from enum import Enum
import math
from typing import TypeVar

from datadiff_osc._canonical import to_primitive


class ReplayValidationError(ValueError):
    """A deterministic fail-closed canonical replay error."""


_T = TypeVar("_T")
_E = TypeVar("_E", bound=Enum)


def _error(path: str, detail: str) -> ReplayValidationError:
    return ReplayValidationError(f"{path}: {detail}")


def require_exact_mapping(
    value: object,
    *,
    fields: Iterable[str],
    path: str,
) -> dict[str, object]:
    """Return a JSON object only when its field set is exactly declared."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise _error(path, "expected a string-keyed object")
    expected = frozenset(fields)
    actual = frozenset(value)
    if actual != expected:
        missing = ",".join(sorted(expected - actual)) or "-"
        extra = ",".join(sorted(actual - expected)) or "-"
        raise _error(path, f"field mismatch (missing={missing}; extra={extra})")
    return value


def require_text(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "expected a non-empty string")
    return value


def require_nonnegative_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(path, "expected a non-negative integer")
    return value


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise _error(path, "expected a boolean")
    return value


def require_finite_number(value: object, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "expected a finite number")
    try:
        finite = math.isfinite(float(value))
    except OverflowError as exc:
        raise _error(path, "expected a finite number") from exc
    if not finite:
        raise _error(path, "expected a finite number")
    return value


def require_tuple(
    value: object,
    *,
    path: str,
    item_replayer: Callable[[object, str], _T] | None = None,
    min_length: int = 0,
    unique: bool = False,
    sorted_values: bool = False,
) -> tuple[_T, ...] | tuple[object, ...]:
    """Replay a canonical sequence as an immutable tuple."""

    if isinstance(min_length, bool) or not isinstance(min_length, int) or min_length < 0:
        raise ValueError("min_length must be a non-negative integer")
    if not isinstance(value, (list, tuple)):
        raise _error(path, "expected a canonical sequence")
    replay = item_replayer or replay_immutable_value
    result = tuple(replay(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) < min_length:
        raise _error(path, f"expected at least {min_length} items")
    if unique and any(item in result[:index] for index, item in enumerate(result)):
        raise _error(path, "items must be unique")
    if sorted_values:
        try:
            ordered = tuple(sorted(result))
        except TypeError as exc:
            raise _error(path, "items are not orderable") from exc
        if result != ordered:
            raise _error(path, "items must be sorted")
    return result


def replay_enum(enum_type: type[_E], value: object, *, path: str) -> _E:
    """Replay only an exact frozen enum value, without coercion."""

    if not isinstance(enum_type, type) or not issubclass(enum_type, Enum):
        raise TypeError("enum_type must be an Enum subclass")
    for member in enum_type:
        if type(value) is type(member.value) and value == member.value:
            return member
    raise _error(path, f"unknown {enum_type.__name__} value")


def _replay_tagged_bytes(value: dict[str, object], *, path: str) -> bytes:
    encoded = value["$bytes"]
    if not isinstance(encoded, str):
        raise _error(path, "$bytes payload must be a string")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise _error(path, "invalid canonical base64 bytes") from exc
    if base64.b64encode(decoded).decode("ascii") != encoded:
        raise _error(path, "non-canonical base64 bytes")
    return decoded


def _replay_tagged_float(value: dict[str, object], *, path: str) -> float:
    encoded = value["$float"]
    if not isinstance(encoded, str):
        raise _error(path, "$float payload must be a string")
    if encoded == "nan":
        decoded = float("nan")
    elif encoded == "+inf":
        decoded = float("inf")
    elif encoded == "-inf":
        decoded = float("-inf")
    elif encoded == "-0":
        decoded = -0.0
    else:
        try:
            decoded = float.fromhex(encoded)
        except ValueError as exc:
            raise _error(path, "invalid canonical hexadecimal float") from exc
    if to_primitive(decoded) != value:
        raise _error(path, "non-canonical tagged float")
    return decoded


def replay_immutable_value(value: object, path: str = "value") -> object:
    """Restore canonical immutable values, including explicitly tagged scalars."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise _error(path, "untagged floats are not canonical")
    if isinstance(value, (list, tuple)):
        return tuple(
            replay_immutable_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, dict):
        if set(value) == {"$bytes"}:
            return _replay_tagged_bytes(value, path=path)
        if set(value) == {"$float"}:
            return _replay_tagged_float(value, path=path)
        raise _error(path, "ambiguous mapping in an immutable value")
    raise _error(path, f"unsupported canonical value type {type(value).__name__}")


def assert_payload_roundtrip(
    payload: object,
    reconstructed: object,
    *,
    path: str = "payload",
) -> None:
    """Require the reconstructed value to reproduce the decoded payload exactly."""

    try:
        replayed = to_primitive(reconstructed)
    except (TypeError, ValueError) as exc:
        raise _error(path, "reconstructed value is not canonicalizable") from exc
    if replayed != payload:
        raise _error(path, "reconstructed value does not round-trip exactly")


__all__ = [
    "ReplayValidationError",
    "assert_payload_roundtrip",
    "replay_enum",
    "replay_immutable_value",
    "require_bool",
    "require_exact_mapping",
    "require_finite_number",
    "require_nonnegative_int",
    "require_text",
    "require_tuple",
]

