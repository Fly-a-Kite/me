from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from datadiff.canonicalization import BatchResultComparison, canonical_key, compare_result_batch
from datadiff.case_features import case_contains_special_float
from datadiff.dsl import Case
from datadiff.logical_types import (
    LOGICAL_DTYPE_SCHEMA_VERSION,
    logical_dtype_token,
    logical_dtype_tokens,
)
from datadiff.normalizer import NormalizedResult
from datadiff.program_analysis import numeric_cast_string_columns
from datadiff.semantic_contracts import semantic_contract_lattice


CONTRACT_COMPARISON_SCHEMA_VERSION = "contract-comparison-v1"
COMPARISON_VIEWS = frozenset(
    {
        "exact",
        "ordered_value",
        "bag_value",
        "numeric_tolerant",
        "error_equivalent",
    }
)


@dataclass(frozen=True, slots=True)
class ContractComparisonProfile:
    view: str
    ordering_policy: str
    null_policy: str
    nan_policy: str
    dtype_policy: str
    error_policy: str
    include_dtype: bool = False
    numeric_decimals: int = 10
    nan_null_equivalent_backends: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRACT_COMPARISON_SCHEMA_VERSION,
            "view": self.view,
            "ordering_policy": self.ordering_policy,
            "null_policy": self.null_policy,
            "nan_policy": self.nan_policy,
            "dtype_policy": self.dtype_policy,
            "error_policy": self.error_policy,
            "include_dtype": self.include_dtype,
            "numeric_decimals": self.numeric_decimals,
            "nan_null_equivalent_backends": list(self.nan_null_equivalent_backends),
            "logical_dtype_schema_version": LOGICAL_DTYPE_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True)
class ContractComparisonResult:
    comparison: BatchResultComparison
    profile: ContractComparisonProfile

    @property
    def mismatch_class(self) -> str:
        return self.comparison.mismatch_class

    @property
    def has_mismatch(self) -> bool:
        return self.comparison.has_mismatch

    def suspicious_labels(self, labels: Sequence[str]) -> list[str]:
        return self.comparison.suspicious_labels(labels)

    def default_confidence(self) -> str:
        return self.comparison.default_confidence()


def comparison_profile_for_case(case: Case) -> ContractComparisonProfile:
    lattice = semantic_contract_lattice(case)
    options = _comparison_options(case)
    requested_view = str(options.get("view", options.get("comparison_view", "")) or "")
    if requested_view and requested_view not in COMPARISON_VIEWS:
        allowed = ", ".join(sorted(COMPARISON_VIEWS))
        raise ValueError(
            f"unknown semantic comparison view {requested_view!r}; expected one of: {allowed}"
        )
    if not requested_view:
        requested_view = "ordered_value" if case.program.output_order_sensitive else "bag_value"
    include_dtype = bool(options.get("include_dtype", requested_view == "exact"))
    try:
        numeric_decimals = max(0, min(18, int(options.get("numeric_decimals", 10))))
    except (TypeError, ValueError):
        numeric_decimals = 10
    configured_nan_null_backends = options.get("nan_null_equivalent_backends")
    if isinstance(configured_nan_null_backends, (list, tuple, set)):
        nan_null_equivalent_backends = tuple(
            str(backend)
            for backend in configured_nan_null_backends
            if str(backend)
        )
    elif (
        requested_view != "exact"
        and not case_contains_special_float(case)
    ):
        # pandas represents nullable scalar values as NaN in several otherwise
        # non-floating result columns. The lossless observation is retained, but
        # the default value view treats this adapter materialization as NULL.
        nan_null_equivalent_backends = ("pandas",)
    else:
        nan_null_equivalent_backends = ()
    return ContractComparisonProfile(
        view=requested_view,
        ordering_policy=lattice.axes["ordering"].policy,
        null_policy=lattice.axes["null"].policy,
        nan_policy=lattice.axes["nan"].policy,
        dtype_policy=lattice.axes["dtype_coercion"].policy,
        error_policy=lattice.axes["error_equivalence"].policy,
        include_dtype=include_dtype,
        numeric_decimals=numeric_decimals,
        nan_null_equivalent_backends=nan_null_equivalent_backends,
    )


def compare_results_under_contract(
    case: Case,
    results: Sequence[NormalizedResult],
    *,
    profile: ContractComparisonProfile | None = None,
) -> ContractComparisonResult:
    resolved_profile = profile or comparison_profile_for_case(case)
    payloads = [
        comparison_payload_for_case(case, result, profile=resolved_profile)
        for result in results
    ]
    comparison = compare_result_batch(payloads)
    if comparison.mismatch_class == "schema" and _is_dtype_only_mismatch(results, resolved_profile):
        comparison = BatchResultComparison(
            group_ids=comparison.group_ids,
            mismatch_class="dtype",
            suspicious_indices_value=comparison.suspicious_indices_value,
            has_clear_majority_value=comparison.has_clear_majority_value,
            majority_group_value=comparison.majority_group_value,
        )
    return ContractComparisonResult(comparison=comparison, profile=resolved_profile)


def comparison_payload_for_case(
    case: Case,
    result: NormalizedResult,
    *,
    profile: ContractComparisonProfile | None = None,
) -> dict[str, Any]:
    resolved = profile or comparison_profile_for_case(case)
    columns = list(result.columns)
    if resolved.include_dtype:
        logical_types = logical_dtype_tokens(result.column_types)
        columns = [
            f"{column}\u0000{logical_types[index] if index < len(logical_types) else 'unknown'}"
            for index, column in enumerate(columns)
        ]
    if result.status != "ok":
        error_type = result.error_type
        if resolved.view == "error_equivalent":
            error_type = _error_category(result.error_type, result.error)
        return {
            "status": result.status,
            "columns": columns,
            "rows": [],
            "error_type": error_type,
        }

    numeric_string_columns = numeric_cast_string_columns(case)
    if resolved.view == "exact":
        rows: list[Any] = [
            [_exact_semantic_value_view(value) for value in row]
            for row in result.effective_lossless_rows
        ]
    else:
        rows = [
            [
                _comparison_value_view(
                    value,
                    column=(
                        result.columns[index]
                        if index < len(result.columns)
                        else ""
                    ),
                    numeric_cast_string_columns=numeric_string_columns,
                    numeric_tolerant=resolved.view == "numeric_tolerant",
                    numeric_decimals=resolved.numeric_decimals,
                    nan_null_equivalent=(
                        result.backend in resolved.nan_null_equivalent_backends
                    ),
                )
                for index, value in enumerate(row)
            ]
            for row in result.effective_lossless_rows
        ]
    if resolved.view in {"bag_value", "numeric_tolerant"}:
        rows = sorted(rows, key=canonical_key)
    return {
        "status": result.status,
        "columns": columns,
        "rows": rows,
        "error_type": result.error_type,
    }


def _comparison_value_view(
    value: Mapping[str, Any],
    *,
    column: str,
    numeric_cast_string_columns: set[str],
    numeric_tolerant: bool,
    numeric_decimals: int,
    nan_null_equivalent: bool,
) -> dict[str, Any]:
    if column in numeric_cast_string_columns and value.get("kind") == "str":
        canonical = _canonical_numeric_string(str(value.get("value", "")))
        if canonical is not None:
            return {"kind": "numeric_string", "value": canonical}
    return _semantic_value_view(
        value,
        numeric_tolerant=numeric_tolerant,
        numeric_decimals=numeric_decimals,
        nan_null_equivalent=nan_null_equivalent,
    )


def _canonical_numeric_string(value: str) -> str | None:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number.is_zero():
        return "-0" if number.is_signed() else "0"
    return str(number.normalize())


def comparison_profile_payload(case: Case) -> dict[str, Any]:
    return comparison_profile_for_case(case).to_dict()


def _semantic_value_view(
    value: Mapping[str, Any],
    *,
    numeric_tolerant: bool,
    numeric_decimals: int,
    nan_null_equivalent: bool,
) -> dict[str, Any]:
    kind = str(value.get("kind", "opaque"))
    if kind == "int":
        return {"kind": "number", "value": str(value.get("value", "0"))}
    if kind == "float":
        numeric = _semantic_float(value)
        negative_zero = bool(value.get("negative_zero", False))
        if negative_zero:
            return {"kind": "number", "value": "-0"}
        if numeric_tolerant:
            numeric = round(numeric, numeric_decimals)
        if numeric.is_integer() and abs(numeric) < 2**53:
            return {"kind": "number", "value": str(int(numeric))}
        return {"kind": "number", "value": numeric.hex()}
    if kind == "nan":
        if nan_null_equivalent:
            return {"kind": "null"}
        return {"kind": "nan"}
    if kind == "infinity":
        return {"kind": "infinity", "sign": int(value.get("sign", 0) or 0)}
    if kind == "decimal":
        return {
            "kind": "decimal",
            "value": str(value.get("value", "")),
            "exponent": int(value.get("exponent", 0) or 0),
        }
    if kind == "mapping":
        return {
            "kind": "mapping",
            "entries": [
                {
                    "key": _semantic_value_view(
                        dict(entry.get("key", {})),
                        numeric_tolerant=numeric_tolerant,
                        numeric_decimals=numeric_decimals,
                        nan_null_equivalent=nan_null_equivalent,
                    ),
                    "value": _semantic_value_view(
                        dict(entry.get("value", {})),
                        numeric_tolerant=numeric_tolerant,
                        numeric_decimals=numeric_decimals,
                        nan_null_equivalent=nan_null_equivalent,
                    ),
                }
                for entry in list(value.get("entries", []) or [])
                if isinstance(entry, Mapping)
            ],
        }
    if kind in {"sequence", "set"}:
        items = [
            _semantic_value_view(
                dict(item),
                numeric_tolerant=numeric_tolerant,
                numeric_decimals=numeric_decimals,
                nan_null_equivalent=nan_null_equivalent,
            )
            for item in list(value.get("items", []) or [])
            if isinstance(item, Mapping)
        ]
        if kind == "set":
            items.sort(key=canonical_key)
        return {"kind": kind, "items": items}
    out = {"kind": kind}
    for key in (
        "value",
        "hex",
        "days",
        "seconds",
        "microseconds",
        "fold",
        "timezone",
        "repr",
    ):
        if key in value:
            out[key] = value[key]
    return out


def _exact_semantic_value_view(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key == "runtime_type":
            continue
        if key == "observed_type":
            out["logical_type"] = logical_dtype_token(item)
            continue
        if isinstance(item, Mapping):
            out[key] = _exact_semantic_value_view(item)
            continue
        if isinstance(item, list):
            out[key] = [
                _exact_semantic_value_view(child) if isinstance(child, Mapping) else child
                for child in item
            ]
            continue
        out[key] = item
    return out


def _semantic_float(value: Mapping[str, Any]) -> float:
    hex_value = str(value.get("hex", "") or "")
    if hex_value:
        try:
            return float.fromhex(hex_value)
        except ValueError:
            pass
    return float(str(value.get("value", "0") or "0"))


def _comparison_options(case: Case) -> Mapping[str, Any]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    options = metadata.get("semantic_comparison", {})
    if isinstance(options, Mapping):
        return options
    return {}


def _is_dtype_only_mismatch(
    results: Sequence[NormalizedResult],
    profile: ContractComparisonProfile,
) -> bool:
    if not profile.include_dtype or not results:
        return False
    columns = {tuple(result.columns) for result in results}
    dtypes = {tuple(logical_dtype_tokens(result.column_types)) for result in results}
    return len(columns) == 1 and len(dtypes) > 1


def _error_category(error_type: str, error: str) -> str:
    text = f"{error_type} {error}".lower()
    categories = (
        ("timeout", ("timeout", "timed out")),
        ("overflow", ("overflow", "out of range")),
        ("divide_by_zero", ("divide by zero", "division by zero")),
        ("not_supported", ("notimplemented", "not implemented", "unsupported")),
        ("missing_name", ("missing column", "not found", "unknown column", "binder")),
        ("invalid_input", ("valueerror", "invalid", "parse", "conversion")),
        ("type_error", ("typeerror", "type mismatch", "cannot cast")),
    )
    for category, tokens in categories:
        if any(token in text for token in tokens):
            return category
    return str(error_type or "error").lower()
