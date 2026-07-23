from __future__ import annotations

from collections.abc import Mapping
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
from datadiff.logical_types import logical_dtype_tokens
from datadiff.semantic_values import (
    LOSSLESS_VALUE_SCHEMA_VERSION,
    encode_semantic_rows,
    encode_semantic_value,
    semantic_rows_signature,
)


@dataclass(slots=True)
class NormalizedResult:
    backend: str
    status: str
    columns: list[str]
    rows: list[list[Any]]
    error_type: str = ""
    error: str = ""
    column_types: list[str] = field(default_factory=list)
    lossless_rows: list[list[dict[str, Any]]] = field(default_factory=list)
    lossless_schema_version: str = ""
    capability_decision: dict[str, Any] | None = None
    _comparison_key_cache: str = field(default="", init=False, repr=False)
    _row_profile_cache: RowSetProfile | None = field(default=None, init=False, repr=False)
    _stable_row_keys_cache: list[str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, backend: str | None = None) -> NormalizedResult:
        columns = [str(column) for column in list(payload.get("columns", []) or [])]
        rows = [list(row) for row in list(payload.get("rows", []) or [])]
        column_types = [str(dtype) for dtype in list(payload.get("column_types", []) or [])]
        lossless_rows = [
            [dict(value) for value in list(row)]
            for row in list(payload.get("lossless_rows", []) or [])
        ]
        result = cls(
            backend=str(payload.get("backend", backend or "")),
            status=str(payload.get("status", "")),
            columns=columns,
            rows=rows,
            error_type=str(payload.get("error_type", "")),
            error=str(payload.get("error", "")),
            column_types=column_types,
            lossless_rows=lossless_rows,
            lossless_schema_version=str(payload.get("lossless_schema_version", "") or ""),
            capability_decision=(
                dict(payload["capability_decision"])
                if isinstance(payload.get("capability_decision"), Mapping)
                else None
            ),
        )
        ordered_row_signature = str(payload.get("ordered_row_signature", "") or "")
        unordered_row_signature = str(payload.get("unordered_row_signature", "") or "")
        has_duplicate_rows = payload.get("has_duplicate_rows")
        if (
            ordered_row_signature
            and unordered_row_signature
            and isinstance(has_duplicate_rows, bool)
        ):
            result._row_profile_cache = RowSetProfile(
                ordered_signature=ordered_row_signature,
                unordered_signature=unordered_row_signature,
                has_duplicates=has_duplicate_rows,
            )
        comparison_key = str(payload.get("comparison_key", "") or "")
        if comparison_key:
            result._comparison_key_cache = comparison_key
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "error_type": self.error_type,
            "error": self.error,
            "column_types": self.column_types,
            "logical_column_types": self.logical_column_types,
            "lossless_rows": self.lossless_rows,
            "lossless_schema_version": self.lossless_schema_version,
            "capability_decision": self.capability_decision,
            "ordered_row_signature": self.ordered_row_signature,
            "unordered_row_signature": self.unordered_row_signature,
            "has_duplicate_rows": self.has_duplicate_rows,
            "comparison_key": self.comparison_key,
        }

    def comparison_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "columns": self.columns,
            "rows": self.rows,
            "error_type": self.error_type,
        }

    def lossless_comparison_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "columns": self.columns,
            "column_types": self.column_types,
            "rows": self.effective_lossless_rows,
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

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def logical_column_types(self) -> list[str]:
        return logical_dtype_tokens(self.column_types)

    @property
    def effective_lossless_rows(self) -> list[list[dict[str, Any]]]:
        if self.lossless_rows:
            return self.lossless_rows
        return encode_semantic_rows(self.rows, column_types=self.column_types)

    @property
    def lossless_ordered_signature(self) -> str:
        return semantic_rows_signature(self.effective_lossless_rows)

    @property
    def lossless_unordered_signature(self) -> str:
        keys = sorted(canonical_key(row) for row in self.effective_lossless_rows)
        return canonical_key(keys)

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
    if v is None:
        return None
    if isinstance(v, str):
        return unicodedata.normalize("NFC", v)
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return _norm_float(v, preserve_float_precision=preserve_float_precision)
    if hasattr(v, "item"):
        try:
            return _norm_value(v.item(), preserve_float_precision=preserve_float_precision)
        except Exception:
            pass
    try:
        import pandas as pd
        missing = pd.isna(v)
        if isinstance(missing, bool):
            if missing:
                # In the default common subset, missing values are normalized to
                # SQL-style NULL. Real NaN semantics should be studied in a
                # dedicated experiment because several engines erase the
                # distinction when nullable numeric columns are materialized.
                return None
        elif bool(missing):
            # In the default common subset, missing values are normalized to
            # SQL-style NULL. Real NaN semantics should be studied in a
            # dedicated experiment because several engines erase the
            # distinction when nullable numeric columns are materialized.
            return None
    except Exception:
        pass
    return v


def _norm_float(value: float, *, preserve_float_precision: bool = False) -> Any:
    value = float(value)
    if math.isnan(value):
        return None
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


def _observed_column_types(data: Any, columns: list[str]) -> list[str]:
    if data is None:
        return ["" for _column in columns]
    native_column_types = getattr(data, "column_types", None)
    if native_column_types is not None:
        try:
            values = [str(dtype) for dtype in list(native_column_types)]
            if len(values) == len(columns):
                return values
        except Exception:
            pass
    schema = getattr(data, "schema", None)
    schema_types = getattr(schema, "types", None)
    if schema_types is not None:
        try:
            values = [str(dtype) for dtype in list(schema_types)]
            if len(values) == len(columns):
                return values
        except Exception:
            pass
    if isinstance(schema, Mapping):
        try:
            return [str(schema.get(column, "")) for column in columns]
        except Exception:
            pass
    dtypes = getattr(data, "dtypes", None)
    if dtypes is not None:
        try:
            values = [str(dtype) for dtype in list(dtypes)]
            if len(values) == len(columns):
                return values
        except Exception:
            pass
    return ["" for _column in columns]


def normalize_result(result: BackendResult, program: Program, enable_normalizer: bool = True) -> NormalizedResult:
    if result.status != "ok":
        return NormalizedResult(
            backend=result.backend,
            status=result.status,
            columns=[],
            rows=[],
            error_type=result.error_type,
            error=result.error[:500],
            capability_decision=result.capability_decision,
        )
    try:
        native_table = _native_rows_and_columns(result.data)
        if native_table is None:
            df = _to_pandas(result.data)
            original_columns = [str(c) for c in list(df.columns)]
            original_column_types = _observed_column_types(result.data, original_columns)
            if not any(original_column_types):
                original_column_types = _observed_column_types(df, original_columns)
            raw_rows_iter = df.itertuples(index=False, name=None)
        else:
            original_columns, raw_rows = native_table
            original_column_types = _observed_column_types(result.data, original_columns)
            raw_rows_iter = raw_rows
        column_positions = sorted(enumerate(original_columns), key=lambda item: (item[1], item[0]))
        columns = [name for _, name in column_positions]
        column_types = [
            original_column_types[index] if index < len(original_column_types) else ""
            for index, _name in column_positions
        ]
        rows: list[list[Any]] = []
        lossless_rows: list[list[dict[str, Any]]] = []
        preserve_float_precision = program.order_sensitive
        for raw_row in raw_rows_iter:
            ordered_raw_values = [raw_row[index] for index, _name in column_positions]
            rows.append(
                [
                    _norm_value(value, preserve_float_precision=preserve_float_precision)
                    for value in ordered_raw_values
                ]
            )
            lossless_rows.append(
                [
                    encode_semantic_value(
                        value,
                        observed_type=column_types[index] if index < len(column_types) else "",
                    )
                    for index, value in enumerate(ordered_raw_values)
                ]
            )
        normalized = NormalizedResult(
            result.backend,
            "ok",
            columns=columns,
            rows=rows,
            column_types=column_types,
            lossless_rows=lossless_rows,
            lossless_schema_version=LOSSLESS_VALUE_SCHEMA_VERSION,
            capability_decision=result.capability_decision,
        )
        if enable_normalizer and not program.output_order_sensitive:
            # SQL/DataFrame backends differ on stable ordering for ties and on
            # whether intermediate order survives operations such as joins.
            # Internal order observers still retain exact floating-point values,
            # but final rows use bag semantics unless the suffix of the program
            # defines an observable order.
            normalized.adopt_canonicalized_rows(canonicalize_rows(rows))
            normalized.lossless_rows = canonicalize_rows(lossless_rows).rows
        return normalized
    except Exception as exc:  # noqa: BLE001
        return NormalizedResult(
            result.backend,
            "normalization_error",
            [],
            [],
            type(exc).__name__,
            str(exc)[:500],
            capability_decision=result.capability_decision,
        )
