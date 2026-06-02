from __future__ import annotations

from datadiff.backends.probe_semantics import (
    EXTENDED_FALSE_PROBE_KINDS,
    is_extended_false_probe_kind,
    resolve_bool_probe,
)


def test_resolve_bool_probe_prefers_handler_result():
    observed = resolve_bool_probe(
        "window_avg_probe",
        handlers={"window_avg_probe": lambda: True},
        fallback_false_kinds=EXTENDED_FALSE_PROBE_KINDS,
    )

    assert observed is True


def test_resolve_bool_probe_falls_back_to_extended_false():
    assert resolve_bool_probe(
        "csv_long_numeric_roundtrip_probe",
        fallback_false_kinds=EXTENDED_FALSE_PROBE_KINDS,
    ) is False


def test_resolve_bool_probe_supports_default_false_set():
    assert resolve_bool_probe("random_case_probe") is False


def test_resolve_bool_probe_returns_none_for_non_probe_kind():
    assert resolve_bool_probe("select") is None


def test_extended_false_probe_registry_contains_shared_sql_and_dataframe_fallbacks():
    assert is_extended_false_probe_kind("group_quantile_probe")
    assert is_extended_false_probe_kind("csv_long_numeric_roundtrip_probe")
