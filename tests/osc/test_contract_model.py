from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from datadiff_osc._canonical import canonical_json
from datadiff_osc.contract_engine import Observation, Verdict, VerdictKind
from datadiff_osc.contract_engine.model import ComponentVerdict, SchemaField


def test_endpoint_observation_contract_and_verdict_digests_are_stable(
    endpoints, simple_compiled, int_schema
):
    observation = Observation.build(
        endpoint_id="left",
        status="ok",
        schema=int_schema,
        rows=[[1], [2]],
        execution_metadata={"b": 2, "a": 1},
    )
    clone = Observation.build(
        endpoint_id="left",
        status="ok",
        schema=int_schema,
        rows=[[1], [2]],
        execution_metadata={"a": 1, "b": 2},
    )

    assert endpoints[0].digest == endpoints[0].digest
    assert observation == clone
    assert observation.digest == clone.digest
    assert simple_compiled.contract.digest == simple_compiled.contract.digest
    json.loads(canonical_json(simple_compiled.contract))


def test_models_are_immutable_and_observation_deep_freezes_rows(int_schema):
    mutable = {"kind": "int", "value": "1"}
    observation = Observation.build(
        endpoint_id="left", status="ok", schema=int_schema, rows=[[mutable]]
    )
    mutable["value"] = "2"

    assert dict(observation.rows[0][0])["value"] == "1"
    with pytest.raises(FrozenInstanceError):
        observation.status = "crash"  # type: ignore[misc]
    with pytest.raises(ValueError, match="deeply immutable"):
        Observation("left", "ok", rows=((["mutable"],),))


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ((ComponentVerdict.build("x", VerdictKind.SATISFIED),), VerdictKind.SATISFIED),
        ((ComponentVerdict.build("x", VerdictKind.INAPPLICABLE),), VerdictKind.INAPPLICABLE),
        ((ComponentVerdict.build("x", VerdictKind.INCONCLUSIVE),), VerdictKind.INCONCLUSIVE),
        ((ComponentVerdict.build("x", VerdictKind.VIOLATED),), VerdictKind.VIOLATED),
        (
            (
                ComponentVerdict.build("a", VerdictKind.INAPPLICABLE),
                ComponentVerdict.build("b", VerdictKind.VIOLATED),
            ),
            VerdictKind.VIOLATED,
        ),
    ],
)
def test_four_valued_verdict_aggregation(components, expected):
    assert Verdict.aggregate(components).kind == expected


def test_observation_rejects_unknown_or_inconsistent_status(int_schema):
    with pytest.raises(ValueError, match="unknown observation status"):
        Observation.build(endpoint_id="x", status="silently_ok", schema=int_schema)
    with pytest.raises(ValueError, match="cannot carry"):
        Observation.build(
            endpoint_id="x",
            status="ok",
            schema=int_schema,
            error_category="type_error",
        )

