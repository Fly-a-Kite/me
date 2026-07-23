from __future__ import annotations

from dataclasses import dataclass

from datadiff_osc._canonical import stable_digest
from datadiff_osc.contract_engine.model import HyperContract, RelationObligation
from datadiff_osc.contract_engine.relations import validate_relation_obligation


@dataclass(frozen=True, slots=True)
class EntailmentGraph:
    edges: tuple[tuple[str, str], ...] = ()
    schema_version: str = "osc-contract-entailment-graph-v1"

    def __post_init__(self) -> None:
        normalized = set(self.edges)
        if len(normalized) != len(self.edges):
            raise ValueError("entailment graph edges must be unique")
        if any(left == right for left, right in normalized):
            raise ValueError("entailment graph cannot contain self edges")
        adjacency: dict[str, set[str]] = {}
        for left, right in normalized:
            if not left or not right:
                raise ValueError("entailment graph nodes must be non-empty")
            adjacency.setdefault(left, set()).add(right)
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("entailment graph must be acyclic")
            if node in visited:
                return
            visiting.add(node)
            for child in adjacency.get(node, ()):
                visit(child)
            visiting.remove(node)
            visited.add(node)

        for node in {item for edge in normalized for item in edge}:
            visit(node)

    @property
    def digest(self) -> str:
        return stable_digest("osc-entailment-graph", self)

    def entails(self, stronger: str, weaker: str, *, strict: bool = False) -> bool:
        if stronger == weaker:
            return not strict
        adjacency: dict[str, set[str]] = {}
        for left, right in self.edges:
            adjacency.setdefault(left, set()).add(right)
        frontier = list(adjacency.get(stronger, ()))
        seen: set[str] = set()
        while frontier:
            node = frontier.pop()
            if node == weaker:
                return True
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(adjacency.get(node, ()))
        return False

    def prove(
        self,
        stronger: RelationObligation,
        weaker: RelationObligation,
    ) -> "EntailmentProof | None":
        """Build a proof only inside an identical endpoint/component scope."""

        safe_relation_edges = {
            ("sequence_equal", "bag_equal"),
            ("numeric_exact", "numeric_tolerant"),
        }
        if (stronger.relation_id, weaker.relation_id) not in safe_relation_edges:
            return None
        stronger_node = stronger.strength or stronger.relation_id
        weaker_node = weaker.strength or weaker.relation_id
        if not self.entails(stronger_node, weaker_node, strict=True):
            # Default compiled strengths are relation IDs; support graph edges
            # written directly over the relation vocabulary as well.
            if not self.entails(
                stronger.relation_id, weaker.relation_id, strict=True
            ):
                return None
        if stronger.endpoint_ids != weaker.endpoint_ids:
            return None
        if not set(weaker.components) <= set(stronger.components):
            return None
        if validate_relation_obligation(stronger) or validate_relation_obligation(weaker):
            return None
        scope_digest = stable_digest(
            "osc-entailment-scope",
            {
                "endpoint_ids": stronger.endpoint_ids,
                "stronger_components": stronger.components,
                "weaker_components": weaker.components,
                "stronger_parameters": stronger.parameters,
                "weaker_parameters": weaker.parameters,
            },
        )
        return EntailmentProof(
            stronger_obligation_id=stronger.obligation_id,
            weaker_obligation_id=weaker.obligation_id,
            graph_digest=self.digest,
            scope_digest=scope_digest,
        )


@dataclass(frozen=True, slots=True)
class EntailmentProof:
    stronger_obligation_id: str
    weaker_obligation_id: str
    graph_digest: str
    scope_digest: str
    schema_version: str = "osc-contract-entailment-proof-v1"

    @property
    def valid(self) -> bool:
        return all(
            (
                self.stronger_obligation_id,
                self.weaker_obligation_id,
                self.graph_digest,
                self.scope_digest,
            )
        ) and self.stronger_obligation_id != self.weaker_obligation_id

    @property
    def digest(self) -> str:
        return stable_digest("osc-entailment-proof", self)


@dataclass(frozen=True, slots=True)
class ContractHypergraph:
    contracts: tuple[HyperContract, ...]
    entailment: EntailmentGraph
    schema_version: str = "osc-contract-hypergraph-v1"

    def __post_init__(self) -> None:
        ids = [item.contract_id for item in self.contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("contract IDs must be unique")

    @property
    def digest(self) -> str:
        return stable_digest("osc-contract-hypergraph", self)


def default_entailment_graph() -> EntailmentGraph:
    return EntailmentGraph(
        edges=(
            ("ordered_exact_dtype", "ordered_exact_value"),
            ("ordered_exact_value", "bag_exact_value"),
            ("bag_exact_value", "set_exact_value"),
            ("exact_error_type", "semantic_error_category"),
            ("numeric_exact", "numeric_tolerant"),
            ("sequence_equal", "bag_equal"),
        )
    )
