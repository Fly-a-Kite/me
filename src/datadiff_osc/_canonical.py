from __future__ import annotations

import base64
from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


CANONICAL_SCHEMA_VERSION = "osc-canonical-json-v1"


def _require_string_mapping(value: Mapping[Any, Any]) -> None:
    non_strings = tuple(type(key).__name__ for key in value if not isinstance(key, str))
    if non_strings:
        raise TypeError(
            "canonical mappings require string keys; non-string keys can collide "
            f"after coercion: {non_strings}"
        )


def to_primitive(value: Any) -> Any:
    """Return a deterministic, JSON-safe representation without losing floats."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float):
        if math.isnan(value):
            return {"$float": "nan"}
        if math.isinf(value):
            return {"$float": "+inf" if value > 0 else "-inf"}
        if value == 0.0 and math.copysign(1.0, value) < 0:
            return {"$float": "-0"}
        return {"$float": value.hex()}
    if isinstance(value, Enum):
        return to_primitive(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_primitive(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        _require_string_mapping(value)
        return {
            key: to_primitive(item)
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        }
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [to_primitive(item) for item in value]
        return sorted(converted, key=canonical_json)
    if hasattr(value, "to_dict"):
        return to_primitive(value.to_dict())
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_primitive(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_digest(namespace: str, value: Any) -> str:
    envelope = {
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "namespace": str(namespace),
        "payload": to_primitive(value),
    }
    return f"{namespace}-" + hashlib.sha256(canonical_json(envelope).encode()).hexdigest()


def canonical_envelope(type_name: str, schema_version: str, value: Any) -> str:
    """Serialize a public value with an explicit type and schema boundary."""

    if not type_name or not schema_version:
        raise ValueError("canonical envelope type and schema version must be non-empty")
    return canonical_json(
        {
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "schema_version": schema_version,
            "type": type_name,
            "payload": to_primitive(value),
        }
    )


def decode_canonical_envelope(payload: str) -> dict[str, Any]:
    """Decode and validate a public envelope without inventing an object registry."""

    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("canonical envelope must decode to an object")
    expected = {"canonical_schema_version", "schema_version", "type", "payload"}
    if set(decoded) != expected:
        raise ValueError("canonical envelope fields do not match the frozen schema")
    if decoded["canonical_schema_version"] != CANONICAL_SCHEMA_VERSION:
        raise ValueError("unsupported canonical schema version")
    if not isinstance(decoded["schema_version"], str) or not decoded["schema_version"]:
        raise ValueError("canonical envelope schema version must be non-empty")
    if not isinstance(decoded["type"], str) or not decoded["type"]:
        raise ValueError("canonical envelope type must be non-empty")
    if canonical_json(decoded) != payload:
        raise ValueError("canonical envelope is not in canonical byte form")
    return decoded


def canonical_roundtrip(payload: str) -> str:
    """Validate and re-encode an envelope byte-for-byte."""

    return canonical_json(decode_canonical_envelope(payload))


def immutable_value(value: Any) -> Any:
    """Deep-freeze common payloads into tuples and immutable scalar values."""

    if value is None or isinstance(value, (str, int, float, bool, bytes, Enum)):
        return value
    if isinstance(value, Mapping):
        _require_string_mapping(value)
        return tuple(
            (key, immutable_value(item))
            for key, item in sorted(value.items(), key=lambda pair: pair[0])
        )
    if isinstance(value, (list, tuple)):
        return tuple(immutable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(
            sorted(
                (immutable_value(item) for item in value),
                key=canonical_json,
            )
        )
    if is_dataclass(value):
        assert_deeply_immutable(value)
        return value
    raise TypeError(f"unsupported immutable value: {type(value).__name__}")


def is_deeply_immutable(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool, bytes, Enum)):
        return True
    if isinstance(value, tuple):
        return all(is_deeply_immutable(item) for item in value)
    if isinstance(value, frozenset):
        return all(is_deeply_immutable(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        return bool(parameters and parameters.frozen) and all(
            is_deeply_immutable(getattr(value, field.name)) for field in fields(value)
        )
    return False


def assert_deeply_immutable(value: Any) -> None:
    if not is_deeply_immutable(value):
        raise TypeError(f"value is not deeply immutable: {type(value).__name__}")


def frozen_pairs(value: Mapping[str, Any] | None = None) -> tuple[tuple[str, Any], ...]:
    _require_string_mapping(value or {})
    return tuple(
        (key, immutable_value(item))
        for key, item in sorted((value or {}).items(), key=lambda pair: pair[0])
    )


def pairs_dict(value: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return {key: item for key, item in value}
