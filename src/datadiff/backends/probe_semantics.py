from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Callable

from datadiff.operation_semantics import is_default_false_probe_kind

EXTENDED_FALSE_PROBE_KINDS: frozenset[str] = frozenset(
    {
        "group_quantile_probe",
        "window_avg_probe",
        "timestamp_precision_filter_probe",
        "series_rtruediv_probe",
        "uint64_isin_probe",
        "float_wrap_probe",
        "empty_literal_groupby_probe",
        "arrow_string_eq_sum_probe",
        "arrow_timestamp_loc_slice_probe",
        "arrow_timestamp_index_attr_probe",
        "bool_reduction_skipna_probe",
        "arrow_bool_groupby_reduction_probe",
        "list_flatten_parent_indices_probe",
        "rolling_mean_by_null_count_probe",
        "csv_long_numeric_roundtrip_probe",
    }
)


def is_extended_false_probe_kind(kind: str) -> bool:
    return kind in EXTENDED_FALSE_PROBE_KINDS


def resolve_bool_probe(
    kind: str,
    *,
    handlers: Mapping[str, Callable[[], bool]] | None = None,
    fallback_false_kinds: Collection[str] = (),
) -> bool | None:
    if handlers is not None:
        handler = handlers.get(kind)
        if handler is not None:
            return bool(handler())
    if kind in fallback_false_kinds or is_default_false_probe_kind(kind):
        return False
    return None
