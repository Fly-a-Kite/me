from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

import datadiff_osc
from datadiff_osc._canonical import to_primitive
from datadiff_osc._replay_support import (
    ReplayValidationError,
    assert_payload_roundtrip,
    replay_enum,
    replay_immutable_value,
    require_bool,
    require_exact_mapping,
    require_finite_number,
    require_nonnegative_int,
    require_text,
    require_tuple,
)
from datadiff_osc.schemas import ExecutionStatus


def test_exact_mapping_rejects_missing_extra_and_non_mapping_values():
    assert require_exact_mapping(
        {"left": 1, "right": 2}, fields={"left", "right"}, path="root"
    ) == {"left": 1, "right": 2}
    with pytest.raises(ReplayValidationError, match=r"missing=right; extra=-"):
        require_exact_mapping({"left": 1}, fields={"left", "right"}, path="root")
    with pytest.raises(ReplayValidationError, match=r"missing=-; extra=other"):
        require_exact_mapping(
            {"left": 1, "right": 2, "other": 3},
            fields={"left", "right"},
            path="root",
        )
    with pytest.raises(ReplayValidationError, match="string-keyed object"):
        require_exact_mapping([], fields=(), path="root")


def test_scalar_requirements_are_exact_and_bool_never_becomes_numeric():
    assert require_text("value", path="text") == "value"
    assert require_nonnegative_int(0, path="count") == 0
    assert require_bool(False, path="flag") is False
    assert require_finite_number(2.5, path="sample") == 2.5
    for value in (True, -1, 1.5, "1"):
        with pytest.raises(ReplayValidationError):
            require_nonnegative_int(value, path="count")
    for value in (0, "false", None):
        with pytest.raises(ReplayValidationError):
            require_bool(value, path="flag")
    for value in (True, float("nan"), float("inf"), "1"):
        with pytest.raises(ReplayValidationError):
            require_finite_number(value, path="sample")


def test_enum_replay_accepts_only_exact_frozen_values():
    assert replay_enum(ExecutionStatus, "ok", path="status") is ExecutionStatus.OK
    with pytest.raises(ReplayValidationError, match="unknown ExecutionStatus"):
        replay_enum(ExecutionStatus, "OK", path="status")
    with pytest.raises(ReplayValidationError, match="unknown ExecutionStatus"):
        replay_enum(ExecutionStatus, 1, path="status")


def test_tuple_replay_deep_freezes_values_and_enforces_identity_rules():
    value = require_tuple(
        ["a", [1, 2], {"$bytes": "AP8="}], path="items", min_length=3
    )
    assert value == ("a", (1, 2), b"\x00\xff")
    with pytest.raises(ReplayValidationError, match="items must be unique"):
        require_tuple(["a", "a"], path="items", unique=True)
    with pytest.raises(ReplayValidationError, match="items must be sorted"):
        require_tuple(["b", "a"], path="items", sorted_values=True)


def test_immutable_replay_restores_only_exact_tagged_scalars():
    values = (
        ({"$float": "0x1.8000000000000p+0"}, 1.5),
        ({"$float": "-0"}, -0.0),
        ({"$float": "+inf"}, float("inf")),
        ({"$float": "nan"}, float("nan")),
        ({"$bytes": "YQ=="}, b"a"),
    )
    for payload, expected in values:
        actual = replay_immutable_value(payload)
        if isinstance(expected, float) and math.isnan(expected):
            assert math.isnan(actual)
        else:
            assert to_primitive(actual) == payload
    for payload in (
        1.5,
        {"$float": "1.5"},
        {"$float": "0x1.8000000000000p+0", "extra": 1},
        {"$bytes": "a"},
        {"caller": "mapping"},
    ):
        with pytest.raises(ReplayValidationError):
            replay_immutable_value(payload)


@dataclass(frozen=True, slots=True)
class _NestedReplayValue:
    name: str
    values: tuple[object, ...]


def test_payload_roundtrip_detects_reconstruction_drift():
    value = _NestedReplayValue("sample", (1, b"a", 1.5))
    payload = to_primitive(value)
    assert_payload_roundtrip(payload, value)
    with pytest.raises(ReplayValidationError, match="does not round-trip"):
        assert_payload_roundtrip({"name": "other", "values": payload["values"]}, value)


def test_private_support_is_not_on_the_frozen_public_surface():
    assert not hasattr(datadiff_osc, "ReplayValidationError")

