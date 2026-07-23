from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff_osc.contract_engine import Observation, VerdictKind
from datadiff_osc.contract_engine.hypergraph import (
    EntailmentGraph,
    default_entailment_graph,
)
from datadiff_osc.contract_engine.model import RelationObligation
from datadiff_osc.contract_engine.monitor import monitor_exact


def _obligation(identity, relation, components=("row_membership",), endpoints=("left", "right")):
    return RelationObligation.build(
        obligation_id=identity,
        relation_id=relation,
        endpoint_ids=endpoints,
        components=components,
        strength=relation,
    )


def test_entailment_graph_rejects_direct_and_indirect_cycles():
    with pytest.raises(ValueError, match="acyclic"):
        EntailmentGraph((('a', 'b'), ('b', 'a')))
    with pytest.raises(ValueError, match="acyclic"):
        EntailmentGraph((('a', 'b'), ('b', 'c'), ('c', 'a')))


def test_arbitrary_graph_edge_cannot_invent_semantic_entailment():
    graph = EntailmentGraph((("bag_equal", "sequence_equal"),))
    assert graph.prove(
        _obligation("bag", "bag_equal"),
        _obligation("sequence", "sequence_equal"),
    ) is None


def test_proof_requires_identical_endpoint_scope_and_component_subset():
    graph = default_entailment_graph()
    strong = _obligation(
        "ordered",
        "sequence_equal",
        components=("row_membership", "presentation_order"),
    )
    weak = _obligation("bag", "bag_equal")
    assert graph.prove(strong, weak) is not None
    assert graph.prove(
        strong,
        _obligation("other", "bag_equal", endpoints=("right", "left")),
    ) is None
    assert graph.prove(
        _obligation("narrow", "sequence_equal"),
        _obligation("wide", "bag_equal", components=("row_membership", "numeric")),
    ) is None


def test_exact_monitor_consumes_only_satisfied_entailment_proof(
    simple_compiled, simple_applicability, endpoints, int_schema
):
    ordered = _obligation(
        "ordered",
        "sequence_equal",
        components=("row_membership", "presentation_order"),
    )
    bag = _obligation("bag", "bag_equal")
    contract = replace(
        simple_compiled.contract,
        contract_id="entailment-contract",
        obligations=(ordered, bag),
    )
    applicability = replace(simple_applicability, contract_digest=contract.digest)
    equal = (
        Observation.build(
            endpoint_id="left", status="ok", schema=int_schema, rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoints[0].digest},
        ),
        Observation.build(
            endpoint_id="right", status="ok", schema=int_schema, rows=[[1], [2]],
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    result = monitor_exact(
        contract,
        equal,
        applicability,
        endpoints=endpoints,
        entailment_graph=default_entailment_graph(),
    )
    assert result.kind == VerdictKind.SATISFIED
    assert "proof-scoped" in result.components[1].reason

    reordered = (
        equal[0],
        Observation.build(
            endpoint_id="right", status="ok", schema=int_schema, rows=[[2], [1]],
            execution_metadata={"endpoint_digest": endpoints[1].digest},
        ),
    )
    result = monitor_exact(
        contract,
        reordered,
        applicability,
        endpoints=endpoints,
        entailment_graph=default_entailment_graph(),
    )
    assert result.components[0].kind == VerdictKind.VIOLATED
    assert result.components[1].kind == VerdictKind.SATISFIED
    assert "proof-scoped" not in result.components[1].reason
