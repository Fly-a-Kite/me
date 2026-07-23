from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any

from datadiff.experiment_manifest import stable_digest


CONTRACT_API_UNIVERSE_SCHEMA_VERSION = "contract-api-universe-v1"
CANONICAL_LOGICAL_TYPES: tuple[str, ...] = (
    "binary",
    "bool",
    "decimal",
    "float",
    "int",
    "nested",
    "str",
    "timestamp",
)

_STRING_OPERATIONS = {
    "filter:str_contains",
    "filter:str_ends_with",
    "filter:str_starts_with",
    "expr:date_part",
    "expr:string_basename",
    "expr:string_concat",
    "expr:string_contains",
    "expr:string_ends_with",
    "expr:string_length",
    "expr:string_lower",
    "expr:string_null_if_empty",
    "expr:string_replace",
    "expr:string_slice",
    "expr:string_split_part",
    "expr:string_starts_with",
    "expr:string_strip",
    "expr:string_upper",
    "op:arrow_string_contains_na_probe",
    "op:arrow_string_eq_sum_probe",
    "op:csv_long_numeric_roundtrip_probe",
    "op:json_predicate_order_probe",
    "op:large_string_partition_probe",
}
_BOOLEAN_OPERATIONS = {
    "agg:all",
    "agg:any",
    "expr:bool_not",
    "op:arrow_bool_groupby_reduction_probe",
    "op:bool_reduction_skipna_probe",
    "op:index_bool_probe",
    "op:sparse_mask_probe",
}
_NUMERIC_OPERATIONS = {
    "agg:mean",
    "agg:sum",
    "expr:abs",
    "expr:add_const",
    "expr:arith_const",
    "expr:clip",
    "expr:reverse_division_columns",
    "op:bit_compare_probe",
    "op:csv_long_numeric_roundtrip_probe",
    "op:float_literal_precision_probe",
    "op:float_wrap_probe",
    "op:group_quantile_probe",
    "op:rolling_mean_by_null_count_probe",
    "op:round_even_probe",
    "op:series_rtruediv_probe",
    "op:datafusion_grouped_null_topk_probe",
    "op:uint64_isin_probe",
    "op:window_avg_probe",
}
_TIMESTAMP_OPERATIONS = {
    "op:arrow_timestamp_index_attr_probe",
    "op:arrow_timestamp_loc_slice_probe",
    "op:polars_timezone_filter_probe",
    "op:timestamp_precision_filter_probe",
}
_NESTED_OPERATIONS = {
    "op:list_flatten_parent_indices_probe",
    "op:struct_distinct_probe",
}
_MULTI_TYPE_OPERATIONS: dict[str, tuple[str, ...]] = {
    "agg:count": CANONICAL_LOGICAL_TYPES,
    "agg:max": ("bool", "decimal", "float", "int", "str", "timestamp"),
    "agg:min": ("bool", "decimal", "float", "int", "str", "timestamp"),
    "agg:nunique": CANONICAL_LOGICAL_TYPES,
    "expr:cast": ("decimal", "float", "int", "str", "timestamp"),
    "op:empty_literal_groupby_probe": ("bool", "float", "int", "str"),
    "op:hash_pivot_wider_probe": ("int", "str"),
    "op:run_end_null_compute_probe": CANONICAL_LOGICAL_TYPES,
    "op:setop_all_duplicate_probe": CANONICAL_LOGICAL_TYPES,
    "op:tuple_anti_null_probe": CANONICAL_LOGICAL_TYPES,
}


def operation_type_domain(operation: str) -> tuple[str, ...]:
    token = str(operation)
    if token in _MULTI_TYPE_OPERATIONS:
        return _MULTI_TYPE_OPERATIONS[token]
    domains: set[str] = set()
    if token in _STRING_OPERATIONS:
        domains.add("str")
    if token in _BOOLEAN_OPERATIONS:
        domains.add("bool")
    if token in _NUMERIC_OPERATIONS:
        domains.update(("decimal", "float", "int"))
    if token in _TIMESTAMP_OPERATIONS:
        domains.add("timestamp")
    if token in _NESTED_OPERATIONS:
        domains.add("nested")
    return tuple(sorted(domains)) if domains else CANONICAL_LOGICAL_TYPES


def operation_type_support(
    operation_tokens: Iterable[str],
    logical_types: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    declared_types = {str(value) for value in logical_types}
    return {
        operation: tuple(
            logical_type
            for logical_type in operation_type_domain(operation)
            if logical_type in declared_types
        )
        for operation in sorted({str(value) for value in operation_tokens})
    }


def build_contract_api_universe(
    targets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    operations = sorted(
        {
            str(operation)
            for target in targets
            for operation in _axis(target.get("capability_model", {}), "operation_tokens")
        }
    )
    cells: list[list[str]] = []
    for target in targets:
        backend = str(target.get("backend", target.get("name", "")) or "")
        model = target.get("capability_model", {})
        if not isinstance(model, Mapping):
            model = {}
        nulls = _axis(model, "null_semantics") or ("",)
        orders = _axis(model, "order_semantics") or ("",)
        modes = _axis(model, "execution_modes") or ("",)
        layouts = _axis(model, "physical_layouts") or ("",)
        for operation in operations:
            for logical_type, null_policy, order_policy, mode, layout in product(
                operation_type_domain(operation),
                nulls,
                orders,
                modes,
                layouts,
            ):
                cells.append(
                    [
                        backend,
                        operation,
                        logical_type,
                        null_policy,
                        order_policy,
                        mode,
                        layout,
                    ]
                )
    payload = {
        "schema_version": CONTRACT_API_UNIVERSE_SCHEMA_VERSION,
        "logical_types": list(CANONICAL_LOGICAL_TYPES),
        "operation_tokens": operations,
        "cell_count": len(cells),
        "cells": cells,
    }
    payload["universe_digest"] = stable_digest("contract-api-universe", payload)
    return payload


def _axis(model: Any, key: str) -> tuple[str, ...]:
    if not isinstance(model, Mapping):
        return ()
    values = model.get(key, ())
    if not isinstance(values, (list, tuple, set, frozenset)):
        return ()
    return tuple(sorted({str(value) for value in values if str(value)}))
