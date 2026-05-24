from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import tempfile
from typing import Any, Iterator

DEFAULT_LONG_NUMERIC_CSV_VALUES = (
    "12345678901234567890",
    "99999999999999999999",
    "9007199254740993",
    "18446744073709551615",
)


def csv_long_numeric_values(op: dict[str, Any] | None = None) -> list[str]:
    raw_values = (op or {}).get("values", DEFAULT_LONG_NUMERIC_CSV_VALUES)
    if not isinstance(raw_values, (list, tuple)):
        return list(DEFAULT_LONG_NUMERIC_CSV_VALUES)
    values = []
    for value in raw_values:
        text = str(value).strip()
        if text and text.isdigit():
            values.append(text)
    return values or list(DEFAULT_LONG_NUMERIC_CSV_VALUES)


@contextmanager
def long_numeric_csv_path(values: list[str]) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "numeric_ids.csv"
        path.write_text(_long_numeric_csv_text(values), encoding="utf-8")
        yield path


def csv_long_numeric_roundtrip_mismatch(observed_values: list[Any], expected_values: list[str]) -> bool:
    observed = [_observed_text(value) for value in observed_values]
    return observed != list(expected_values)


def _long_numeric_csv_text(values: list[str]) -> str:
    rows = ["value,name"]
    rows.extend(f"{value},id_{idx}" for idx, value in enumerate(values))
    return "\n".join(rows) + "\n"


def _observed_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
