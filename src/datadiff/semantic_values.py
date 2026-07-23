from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import json
import math
import numbers
import struct
from typing import Any

from datadiff.canonicalization import canonical_key


LOSSLESS_VALUE_SCHEMA_VERSION = "lossless-semantic-value-v1"


def runtime_type_name(value: Any) -> str:
    typ = type(value)
    module = str(getattr(typ, "__module__", "") or "")
    qualname = str(getattr(typ, "__qualname__", getattr(typ, "__name__", "")) or "")
    if not module or module == "builtins":
        return qualname
    return f"{module}.{qualname}"


def encode_semantic_value(value: Any, *, observed_type: str = "") -> dict[str, Any]:
    runtime_type = runtime_type_name(value)
    metadata = _type_metadata(runtime_type=runtime_type, observed_type=observed_type)

    if value is None:
        return {"kind": "null", **metadata}
    if _is_pandas_nat(value):
        return {"kind": "nat", **metadata}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value, **metadata}
    if isinstance(value, Decimal):
        decimal_tuple = value.as_tuple()
        return {
            "kind": "decimal",
            "value": str(value),
            "sign": int(decimal_tuple.sign),
            "digits": "".join(str(digit) for digit in decimal_tuple.digits),
            "exponent": int(decimal_tuple.exponent),
            **metadata,
        }
    if isinstance(value, numbers.Integral):
        return {
            "kind": "int",
            "value": str(int(value)),
            **metadata,
        }
    if isinstance(value, numbers.Real):
        return _encode_float(value, metadata=metadata)
    if isinstance(value, datetime):
        return {
            "kind": "datetime",
            "value": value.isoformat(),
            "fold": int(getattr(value, "fold", 0)),
            "timezone": str(value.tzinfo) if value.tzinfo is not None else "",
            **metadata,
        }
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat(), **metadata}
    if isinstance(value, time):
        return {
            "kind": "time",
            "value": value.isoformat(),
            "fold": int(getattr(value, "fold", 0)),
            "timezone": str(value.tzinfo) if value.tzinfo is not None else "",
            **metadata,
        }
    if isinstance(value, timedelta):
        return {
            "kind": "timedelta",
            "days": int(value.days),
            "seconds": int(value.seconds),
            "microseconds": int(value.microseconds),
            **metadata,
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value, **metadata}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "kind": "binary",
            "hex": bytes(value).hex(),
            **metadata,
        }
    if isinstance(value, Mapping):
        return {
            "kind": "mapping",
            "entries": [
                {
                    "key": encode_semantic_value(key),
                    "value": encode_semantic_value(item),
                }
                for key, item in value.items()
            ],
            **metadata,
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return {
            "kind": "sequence",
            "sequence_type": runtime_type,
            "items": [encode_semantic_value(item) for item in value],
            **metadata,
        }
    if isinstance(value, (set, frozenset)):
        items = [encode_semantic_value(item) for item in value]
        items.sort(key=canonical_key)
        return {
            "kind": "set",
            "items": items,
            **metadata,
        }
    if _is_scalar_missing(value):
        return {"kind": "null", **metadata}

    converted = _native_scalar_value(value)
    if converted is not value:
        encoded = encode_semantic_value(converted, observed_type=observed_type)
        encoded["runtime_type"] = runtime_type
        return encoded

    return {
        "kind": "opaque",
        "repr": _stable_repr(value),
        **metadata,
    }


def encode_semantic_rows(
    rows: Sequence[Sequence[Any]],
    *,
    column_types: Sequence[str] | None = None,
) -> list[list[dict[str, Any]]]:
    observed_types = list(column_types or ())
    return [
        [
            encode_semantic_value(
                value,
                observed_type=observed_types[index] if index < len(observed_types) else "",
            )
            for index, value in enumerate(row)
        ]
        for row in rows
    ]


def semantic_rows_signature(rows: Sequence[Sequence[Mapping[str, Any]]]) -> str:
    return canonical_key(list(rows))


def semantic_values_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return canonical_key(dict(left)) == canonical_key(dict(right))


def _encode_float(value: Any, *, metadata: dict[str, str]) -> dict[str, Any]:
    numeric = float(value)
    raw_hex, bit_width = _float_raw_hex(value, numeric)
    if math.isnan(numeric):
        return {
            "kind": "nan",
            "raw_hex": raw_hex,
            "bit_width": bit_width,
            **metadata,
        }
    if math.isinf(numeric):
        return {
            "kind": "infinity",
            "sign": 1 if numeric > 0 else -1,
            "raw_hex": raw_hex,
            "bit_width": bit_width,
            **metadata,
        }
    return {
        "kind": "float",
        "value": repr(numeric),
        "hex": numeric.hex(),
        "raw_hex": raw_hex,
        "bit_width": bit_width,
        "negative_zero": bool(numeric == 0.0 and math.copysign(1.0, numeric) < 0.0),
        **metadata,
    }


def _float_raw_hex(value: Any, numeric: float) -> tuple[str, int]:
    dtype = getattr(value, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    tobytes = getattr(value, "tobytes", None)
    if callable(tobytes) and itemsize:
        try:
            return bytes(tobytes()).hex(), int(itemsize) * 8
        except Exception:
            pass
    return struct.pack("!d", numeric).hex(), 64


def _native_scalar_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        try:
            return as_py()
        except Exception:
            pass
    return value


def _is_scalar_missing(value: Any) -> bool:
    try:
        import pandas as pd

        missing = pd.isna(value)
        return isinstance(missing, bool) and missing
    except Exception:
        return False


def _is_pandas_nat(value: Any) -> bool:
    return runtime_type_name(value) in {"pandas._libs.tslibs.nattype.NaTType", "NaTType"}


def _stable_repr(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _type_metadata(*, runtime_type: str, observed_type: str) -> dict[str, str]:
    metadata = {"runtime_type": runtime_type}
    if observed_type:
        metadata["observed_type"] = str(observed_type)
    return metadata
