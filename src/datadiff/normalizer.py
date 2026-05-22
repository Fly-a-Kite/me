from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from typing import Any

from datadiff.backends.base import BackendResult
from datadiff.dsl import Program


@dataclass(slots=True)
class NormalizedResult:
    backend: str
    status: str
    columns: list[str]
    rows: list[list[Any]]
    error_type: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "columns": self.columns,
            "rows": self.rows,
            "error_type": self.error_type,
            "error": self.error,
        }


def _norm_value(v: Any) -> Any:
    try:
        import pandas as pd
        if pd.isna(v):
            # In the default common subset, missing values are normalized to
            # SQL-style NULL. Real NaN semantics should be studied in a
            # dedicated experiment because several engines erase the
            # distinction when nullable numeric columns are materialized.
            return None
    except Exception:
        pass
    if isinstance(v, float):
        value = float(v)
        if math.isnan(value):
            return {"kind": "nan"}
        if math.isinf(value):
            return {"kind": "inf", "sign": 1 if value > 0 else -1}
        if value.is_integer() and abs(value) < 2**53:
            return int(value)
        rounded = float(round(value, 10))
        if rounded.is_integer() and abs(rounded) < 2**53:
            return int(rounded)
        return rounded
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if hasattr(v, "item"):
        try:
            return _norm_value(v.item())
        except Exception:
            pass
    return v


def _to_pandas(data: Any):
    import pandas as pd
    if data is None:
        return pd.DataFrame()
    # Polars can convert through Arrow, but that path may require optional
    # dependencies. For normalization, rows/columns are enough.
    if hasattr(data, "rows") and hasattr(data, "columns"):
        return pd.DataFrame(data.rows(named=True), columns=list(data.columns))
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    if hasattr(data, "to_pandas_dataframe"):
        return data.to_pandas_dataframe()
    return data


def normalize_result(result: BackendResult, program: Program, enable_normalizer: bool = True) -> NormalizedResult:
    if result.status != "ok":
        return NormalizedResult(
            backend=result.backend,
            status=result.status,
            columns=[],
            rows=[],
            error_type=result.error_type,
            error=result.error[:500],
        )
    try:
        df = _to_pandas(result.data)
        original_columns = [str(c) for c in list(df.columns)]
        column_positions = sorted(enumerate(original_columns), key=lambda item: (item[1], item[0]))
        columns = [name for _, name in column_positions]
        rows: list[list[Any]] = []
        for _, row in df.iterrows():
            rows.append([_norm_value(row.iloc[idx]) for idx, _ in column_positions])
        if enable_normalizer and not program.order_sensitive:
            # SQL/DataFrame backends differ on stable ordering for ties and on
            # whether intermediate order is observable. The default oracle is
            # bag-semantics; order-sensitive metamorphic checks should be tested
            # separately with explicit tie-breakers.
            rows = sorted(rows, key=_row_sort_key)
        return NormalizedResult(result.backend, "ok", columns=columns, rows=rows)
    except Exception as exc:  # noqa: BLE001
        return NormalizedResult(result.backend, "normalization_error", [], [], type(exc).__name__, str(exc)[:500])


def _row_sort_key(row: list[Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)
