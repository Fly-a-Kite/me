from __future__ import annotations

from collections.abc import Mapping
import json
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from datadiff.backends.base import BackendResult
from datadiff.canonicalization import (
    CanonicalizedRows,
    RowSetProfile,
    canonical_key,
    canonical_keys,
    canonicalize_rows,
    profile_rows,
    result_comparison_key,
)
from datadiff.dsl import Program


@dataclass(slots=True)
class NormalizedResult:
    backend: str
    status: str
    columns: list[str]
    rows: list[list[Any]]
    error_type: str = ""
    error: str = ""
    _comparison_key_cache: str = field(default="", init=False, repr=False)
    _row_profile_cache: RowSetProfile | None = field(default=None, init=False, repr=False)
    _stable_row_keys_cache: list[str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, backend: str | None = None) -> NormalizedResult:
        columns = [str(column) for column in list(payload.get("columns", []) or [])]
        rows = [list(row) for row in list(payload.get("rows", []) or [])]
        return cls(
            backend=str(payload.get("backend", backend or "")),
            status=str(payload.get("status", "")),
            columns=columns,
            rows=rows,
            error_type=str(payload.get("error_type", "")),
            error=str(payload.get("error", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "columns": self.columns,
            "rows": self.rows,
            "error_type": self.error_type,
            "error": self.error,
        }

    def comparison_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "columns": self.columns,
            "rows": self.rows,
            "error_type": self.error_type,
        }

    @property
    def comparison_key(self) -> str:
        if not self._comparison_key_cache:
            self._comparison_key_cache = result_comparison_key(
                status=self.status,
                columns=self.columns,
                ordered_row_signature=self.ordered_row_signature,
                error_type=self.error_type,
            )
        return self._comparison_key_cache

    @property
    def row_profile(self) -> RowSetProfile:
        if self._row_profile_cache is None:
            self._row_profile_cache = profile_rows(self.rows)
        return self._row_profile_cache

    @property
    def stable_row_keys(self) -> list[str]:
        if self._stable_row_keys_cache is None:
            self._stable_row_keys_cache = canonical_keys(self.rows)
        return self._stable_row_keys_cache

    def adopt_canonicalized_rows(self, canonicalized: CanonicalizedRows) -> None:
        self.rows = canonicalized.rows
        self._stable_row_keys_cache = list(canonicalized.stable_row_keys)
        self._row_profile_cache = canonicalized.profile
        self._comparison_key_cache = ""

    @property
    def ordered_row_signature(self) -> str:
        return self.row_profile.ordered_signature

    @property
    def unordered_row_signature(self) -> str:
        return self.row_profile.unordered_signature

    @property
    def has_duplicate_rows(self) -> bool:
        return self.row_profile.has_duplicates


def normalized_results_from_mapping(
    mapping: Mapping[str, Mapping[str, Any] | NormalizedResult],
) -> dict[str, NormalizedResult]:
    out: dict[str, NormalizedResult] = {}
    for backend, payload in mapping.items():
        key = str(backend)
        if isinstance(payload, NormalizedResult):
            out[key] = payload
            continue
        out[key] = NormalizedResult.from_dict(payload, backend=key)
    return out


def _norm_value(v: Any, *, preserve_float_precision: bool = False) -> Any:
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
        if preserve_float_precision:
            return value
        rounded = float(round(value, 10))
        if rounded.is_integer() and abs(rounded) < 2**53:
            return int(rounded)
        return rounded
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if hasattr(v, "item"):
        try:
            return _norm_value(v.item(), preserve_float_precision=preserve_float_precision)
        except Exception:
            pass
    return v


def _to_pandas(data: Any):
    import pandas as pd
    if data is None:
        return pd.DataFrame()
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    if hasattr(data, "to_pandas_dataframe"):
        return data.to_pandas_dataframe()
    return data


def _native_rows_and_columns(data: Any) -> tuple[list[str], list[list[Any]]] | None:
    if not hasattr(data, "rows") or not hasattr(data, "columns"):
        return None
    columns = [str(column) for column in list(data.columns)]
    try:
        native_rows = data.rows(named=False)
    except TypeError:
        native_rows = data.rows()
    return columns, [list(row_values) for row_values in native_rows]


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
        native_table = _native_rows_and_columns(result.data)
        if native_table is None:
            df = _to_pandas(result.data)
            original_columns = [str(c) for c in list(df.columns)]
            raw_rows = [[row.iloc[idx] for idx in range(len(original_columns))] for _, row in df.iterrows()]
        else:
            original_columns, raw_rows = native_table
        column_positions = sorted(enumerate(original_columns), key=lambda item: (item[1], item[0]))
        columns = [name for _, name in column_positions]
        rows: list[list[Any]] = []
        preserve_float_precision = program.order_sensitive
        for raw_row in raw_rows:
            rows.append(
                [
                    _norm_value(raw_row[idx], preserve_float_precision=preserve_float_precision)
                    for idx, _ in column_positions
                ]
            )
        normalized = NormalizedResult(result.backend, "ok", columns=columns, rows=rows)
        if enable_normalizer and not program.order_sensitive:
            # SQL/DataFrame backends differ on stable ordering for ties and on
            # whether intermediate order is observable. The default oracle is
            # bag-semantics; order-sensitive metamorphic checks should be tested
            # separately with explicit tie-breakers.
            normalized.adopt_canonicalized_rows(canonicalize_rows(rows))
        return normalized
    except Exception as exc:  # noqa: BLE001
        return NormalizedResult(result.backend, "normalization_error", [], [], type(exc).__name__, str(exc)[:500])
