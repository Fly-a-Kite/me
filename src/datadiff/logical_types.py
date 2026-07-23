from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence


LOGICAL_DTYPE_SCHEMA_VERSION = "logical-dtype-v1"


@dataclass(frozen=True, slots=True)
class LogicalDType:
    kind: str
    bit_width: int | None = None
    signed: bool | None = None
    precision: int | None = None
    scale: int | None = None
    unit: str = ""
    timezone: str = ""
    storage: str = ""

    @property
    def token(self) -> str:
        fields = [self.kind]
        if self.bit_width is not None:
            fields.append(str(self.bit_width))
        if self.signed is not None:
            fields.append("signed" if self.signed else "unsigned")
        if self.precision is not None:
            fields.append(f"p{self.precision}")
        if self.scale is not None:
            fields.append(f"s{self.scale}")
        if self.unit:
            fields.append(self.unit)
        if self.timezone:
            fields.append(f"tz={self.timezone}")
        if self.storage:
            fields.append(f"storage={self.storage}")
        return ":".join(fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LOGICAL_DTYPE_SCHEMA_VERSION,
            "kind": self.kind,
            "bit_width": self.bit_width,
            "signed": self.signed,
            "precision": self.precision,
            "scale": self.scale,
            "unit": self.unit,
            "timezone": self.timezone,
            "storage": self.storage,
            "token": self.token,
        }


def parse_logical_dtype(dtype: Any) -> LogicalDType:
    raw = str(dtype or "").strip()
    if not raw:
        return LogicalDType("unknown")
    normalized = _normalize(raw)

    extension_scalar = re.fullmatch(
        r"(u?int(?:8|16|32|64)|float(?:16|32|64)|double|bool|boolean)\[([^\]]+)\]",
        normalized,
    )
    if extension_scalar:
        base, storage = extension_scalar.groups()
        if base.startswith("uint"):
            return LogicalDType(
                "integer",
                bit_width=int(base.removeprefix("uint")),
                signed=False,
                storage=storage,
            )
        if base.startswith("int"):
            return LogicalDType(
                "integer",
                bit_width=int(base.removeprefix("int")),
                signed=True,
                storage=storage,
            )
        if base.startswith("float"):
            return LogicalDType(
                "float",
                bit_width=int(base.removeprefix("float")),
                storage=storage,
            )
        if base == "double":
            return LogicalDType("float", bit_width=64, storage=storage)
        return LogicalDType("boolean", storage=storage)

    integer = re.fullmatch(r"(u?)int(8|16|32|64)", normalized)
    if integer:
        return LogicalDType(
            "integer",
            bit_width=int(integer.group(2)),
            signed=not bool(integer.group(1)),
        )
    sql_integer = {
        "tinyint": 8,
        "smallint": 16,
        "integer": 32,
        "int": 32,
        "bigint": 64,
        "hugeint": 128,
        "utinyint": 8,
        "usmallint": 16,
        "uinteger": 32,
        "ubigint": 64,
    }
    if normalized in sql_integer:
        return LogicalDType(
            "integer",
            bit_width=sql_integer[normalized],
            signed=not normalized.startswith("u"),
        )

    floating = re.fullmatch(r"float(16|32|64)", normalized)
    if floating:
        return LogicalDType("float", bit_width=int(floating.group(1)))
    if normalized in {"double", "doubleprecision"}:
        return LogicalDType("float", bit_width=64)
    if normalized in {"float", "real", "single"}:
        return LogicalDType("float", bit_width=32 if normalized in {"real", "single"} else None)

    if normalized in {"bool", "boolean"}:
        return LogicalDType("boolean")
    if normalized in {"str", "string", "stringview", "utf8", "varchar", "text"}:
        return LogicalDType("string")
    if normalized.startswith("string["):
        storage = normalized.removeprefix("string[").removesuffix("]")
        return LogicalDType("string", storage=storage)
    if normalized in {"largestring", "largeutf8"}:
        return LogicalDType("large_string")
    if normalized in {"binary", "binaryview", "bytes", "bytea"}:
        return LogicalDType("binary")
    if normalized in {"largebinary"}:
        return LogicalDType("large_binary")

    decimal_match = re.search(r"decimal(?:128|256)?\((\d+)\s*,\s*(-?\d+)\)", normalized)
    if decimal_match:
        return LogicalDType(
            "decimal",
            precision=int(decimal_match.group(1)),
            scale=int(decimal_match.group(2)),
        )
    if normalized.startswith("decimal") or normalized.startswith("numeric"):
        return LogicalDType("decimal")

    if normalized in {"date", "date32", "date64"}:
        return LogicalDType("date", unit=normalized.removeprefix("date") or "day")
    if normalized.startswith("time") and not normalized.startswith("timestamp"):
        return LogicalDType("time", unit=_time_unit(normalized))
    if normalized.startswith("timestamp") or normalized.startswith("datetime"):
        return LogicalDType(
            "timestamp",
            unit=_time_unit(normalized),
            timezone=_timezone(normalized),
        )
    if normalized.startswith("duration") or normalized.startswith("timedelta"):
        return LogicalDType("duration", unit=_time_unit(normalized))

    if normalized.startswith("dictionary") or normalized.startswith("categorical") or normalized == "category":
        return LogicalDType("dictionary")
    if normalized.startswith("runendencoded") or normalized.startswith("run_end_encoded"):
        return LogicalDType("run_end_encoded")
    if normalized.startswith("largelist"):
        return LogicalDType("large_list")
    if normalized.startswith("list") or normalized.startswith("array"):
        return LogicalDType("list")
    if normalized.startswith("struct"):
        return LogicalDType("struct")
    if normalized.startswith("map"):
        return LogicalDType("map")
    if normalized in {"null", "none", "nonetype"}:
        return LogicalDType("null")
    if normalized in {"object", "pyobject"}:
        return LogicalDType("opaque", storage="object")
    return LogicalDType("extension", storage=normalized)


def logical_dtype_token(dtype: Any) -> str:
    return parse_logical_dtype(dtype).token


def logical_dtype_tokens(dtypes: Sequence[Any]) -> list[str]:
    return [logical_dtype_token(dtype) for dtype in dtypes]


def _normalize(raw: str) -> str:
    return re.sub(r"\s+", "", raw.strip().lower()).replace("_", "")


def _time_unit(normalized: str) -> str:
    bracket = re.search(r"\[(s|ms|us|ns)(?:,|\])", normalized)
    if bracket:
        return bracket.group(1)
    quoted = re.search(r"timeunit=['\"]?(s|ms|us|ns)", normalized)
    if quoted:
        return quoted.group(1)
    direct = re.search(r"(?:timestamp|datetime|duration|timedelta|time)(s|ms|us|ns)\b", normalized)
    if direct:
        return direct.group(1)
    return ""


def _timezone(normalized: str) -> str:
    bracket = re.search(r"tz=([^\],)]+)", normalized)
    if bracket:
        return bracket.group(1).strip("'\"")
    polars = re.search(r"timezone=['\"]([^'\"]+)['\"]", normalized)
    if polars:
        return polars.group(1)
    return ""
