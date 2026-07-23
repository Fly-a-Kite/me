from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

from datadiff.canonicalization import canonical_key, short_canonical_hash
from datadiff.targets import TargetSpec, target_specs


EXECUTION_LATTICE_SCHEMA_VERSION = "semantic-execution-lattice-v1"


@dataclass(frozen=True, slots=True)
class ExecutionNode:
    backend: str
    family: str
    api_lowering: str
    execution_mode: str
    input_layout: str = "native"
    optimizer_config: tuple[tuple[str, str], ...] = ()
    version: str = ""
    adapter: str = ""

    @classmethod
    def build(
        cls,
        *,
        backend: str,
        family: str,
        api_lowering: str,
        execution_mode: str,
        input_layout: str = "native",
        optimizer_config: Mapping[str, Any] | None = None,
        version: str = "",
        adapter: str = "",
    ) -> "ExecutionNode":
        config = tuple(
            sorted(
                (str(key), canonical_key(value))
                for key, value in (optimizer_config or {}).items()
            )
        )
        return cls(
            backend=str(backend),
            family=str(family),
            api_lowering=str(api_lowering),
            execution_mode=str(execution_mode),
            input_layout=str(input_layout or "native"),
            optimizer_config=config,
            version=str(version or ""),
            adapter=str(adapter or ""),
        )

    @property
    def node_id(self) -> str:
        return f"node-{short_canonical_hash(self.identity_payload(), 16)}"

    def identity_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "family": self.family,
            "api_lowering": self.api_lowering,
            "execution_mode": self.execution_mode,
            "input_layout": self.input_layout,
            "optimizer_config": [list(item) for item in self.optimizer_config],
            "version": self.version,
            "adapter": self.adapter,
        }

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class ExecutionEdge:
    source: str
    target: str
    relation: str
    expected_relation: str = "contract_equivalent"
    preconditions: tuple[str, ...] = ("contract_meet_exists",)
    fault_models: tuple[str, ...] = ()
    coverage_tokens: tuple[str, ...] = ()
    priority: float = 1.0

    @property
    def edge_id(self) -> str:
        return f"edge-{short_canonical_hash(self.identity_payload(), 16)}"

    @property
    def node_ids(self) -> tuple[str, str]:
        return (self.source, self.target)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "expected_relation": self.expected_relation,
            "preconditions": list(self.preconditions),
            "fault_models": list(self.fault_models),
            "coverage_tokens": list(self.coverage_tokens),
            "priority": float(self.priority),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"edge_id": self.edge_id, **self.identity_payload()}


@dataclass(frozen=True, slots=True)
class ExecutionCacheKey:
    case_digest: str
    node_id: str
    environment_digest: str
    adapter_revision: str = ""

    @property
    def key(self) -> str:
        return f"exec-{short_canonical_hash(self.to_dict(), 24)}"

    def to_dict(self) -> dict[str, str]:
        return {
            "case_digest": self.case_digest,
            "node_id": self.node_id,
            "environment_digest": self.environment_digest,
            "adapter_revision": self.adapter_revision,
        }


@dataclass(slots=True)
class SemanticExecutionLattice:
    nodes: dict[str, ExecutionNode] = field(default_factory=dict)
    edges: list[ExecutionEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if len(self.nodes) != len(set(self.nodes)):
            raise ValueError("execution lattice contains duplicate node identifiers")
        for node_id, node in self.nodes.items():
            if node_id != node.node_id:
                raise ValueError(f"node mapping key does not match node identity: {node_id}")
        edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            if edge.source == edge.target:
                raise ValueError(f"edge {edge.edge_id} must connect distinct nodes")
            if edge.edge_id in edge_ids:
                raise ValueError(f"duplicate execution edge: {edge.edge_id}")
            edge_ids.add(edge.edge_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXECUTION_LATTICE_SCHEMA_VERSION,
            "nodes": [self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)],
            "edges": [edge.to_dict() for edge in sorted(self.edges, key=lambda item: item.edge_id)],
        }


@dataclass(frozen=True, slots=True)
class LatticeSelection:
    selected_node_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]
    covered_tokens: tuple[str, ...]
    total_cost: float
    budget: float
    omitted_edge_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_node_ids": list(self.selected_node_ids),
            "selected_edge_ids": list(self.selected_edge_ids),
            "covered_tokens": list(self.covered_tokens),
            "total_cost": self.total_cost,
            "budget": self.budget,
            "omitted_edge_ids": list(self.omitted_edge_ids),
        }


def build_target_execution_lattice(
    backends: Sequence[str],
    *,
    versions: Mapping[str, str] | None = None,
    input_layouts: Mapping[str, str] | None = None,
    optimizer_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> SemanticExecutionLattice:
    specs = target_specs(list(backends))
    nodes: dict[str, ExecutionNode] = {}
    node_specs: dict[str, TargetSpec] = {}
    for spec in specs:
        node = ExecutionNode.build(
            backend=spec.backend,
            family=spec.family,
            api_lowering=spec.execution_model,
            execution_mode=_execution_mode(spec),
            input_layout=(input_layouts or {}).get(spec.backend, "native"),
            optimizer_config=(optimizer_configs or {}).get(spec.backend, {}),
            version=(versions or {}).get(spec.backend, ""),
            adapter=spec.adapter,
        )
        nodes[node.node_id] = node
        node_specs[node.node_id] = spec

    edges: list[ExecutionEdge] = []
    for left_id, right_id in combinations(sorted(nodes), 2):
        left = nodes[left_id]
        right = nodes[right_id]
        left_spec = node_specs[left_id]
        right_spec = node_specs[right_id]
        relation, priority = _edge_relation(left_spec, right_spec)
        family_pair = "|".join(sorted((left.family, right.family)))
        mode_pair = "|".join(sorted((left.execution_mode, right.execution_mode)))
        layout_pair = "|".join(sorted((left.input_layout, right.input_layout)))
        edges.append(
            ExecutionEdge(
                source=left_id,
                target=right_id,
                relation=relation,
                fault_models=(
                    f"family_transition:{family_pair}",
                    f"mode_transition:{mode_pair}",
                    f"layout_transition:{layout_pair}",
                ),
                coverage_tokens=(
                    f"relation:{relation}",
                    f"backend_pair:{'|'.join(sorted((left.backend, right.backend)))}",
                    f"family_pair:{family_pair}",
                    f"mode_pair:{mode_pair}",
                    f"layout_pair:{layout_pair}",
                ),
                priority=priority,
            )
        )
    return SemanticExecutionLattice(nodes=nodes, edges=edges)


def select_budgeted_subgraph(
    lattice: SemanticExecutionLattice,
    *,
    budget: float,
    node_costs: Mapping[str, float] | None = None,
    anchor_node_ids: Sequence[str] = (),
    already_covered_tokens: Sequence[str] = (),
    token_weights: Mapping[str, float] | None = None,
) -> LatticeSelection:
    lattice.validate()
    resolved_budget = max(0.0, float(budget))
    costs = {
        node_id: max(0.0, float((node_costs or {}).get(node_id, 1.0)))
        for node_id in lattice.nodes
    }
    selected_nodes = {str(node_id) for node_id in anchor_node_ids}
    unknown_anchors = selected_nodes - set(lattice.nodes)
    if unknown_anchors:
        raise ValueError(f"unknown anchor nodes: {', '.join(sorted(unknown_anchors))}")
    total_cost = sum(costs[node_id] for node_id in selected_nodes)
    if total_cost > resolved_budget:
        raise ValueError("anchor node cost exceeds execution-lattice budget")
    covered = {str(token) for token in already_covered_tokens}
    weights = {str(token): max(0.0, float(weight)) for token, weight in (token_weights or {}).items()}
    selected_edges: list[ExecutionEdge] = []
    remaining = list(lattice.edges)

    while remaining:
        candidates: list[tuple[float, float, float, str, ExecutionEdge, set[str]]] = []
        for edge in remaining:
            edge_nodes = set(edge.node_ids)
            marginal_nodes = edge_nodes - selected_nodes
            marginal_cost = sum(costs[node_id] for node_id in marginal_nodes)
            if total_cost + marginal_cost > resolved_budget:
                continue
            new_tokens = set(edge.coverage_tokens) - covered
            coverage_gain = sum(weights.get(token, 1.0) for token in new_tokens)
            gain = max(0.0, float(edge.priority)) + coverage_gain
            score = gain / max(marginal_cost, 1e-9)
            candidates.append(
                (score, gain, marginal_cost, edge.edge_id, edge, new_tokens)
            )
        if not candidates:
            break
        candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
        _score, _gain, marginal_cost, _edge_id, selected, new_tokens = candidates[0]
        selected_edges.append(selected)
        selected_nodes.update(selected.node_ids)
        covered.update(new_tokens)
        total_cost += marginal_cost
        remaining = [edge for edge in remaining if edge.edge_id != selected.edge_id]

    selected_edge_ids = {edge.edge_id for edge in selected_edges}
    return LatticeSelection(
        selected_node_ids=tuple(sorted(selected_nodes)),
        selected_edge_ids=tuple(edge.edge_id for edge in selected_edges),
        covered_tokens=tuple(sorted(covered)),
        total_cost=total_cost,
        budget=resolved_budget,
        omitted_edge_ids=tuple(
            sorted(edge.edge_id for edge in lattice.edges if edge.edge_id not in selected_edge_ids)
        ),
    )


def _execution_mode(spec: TargetSpec) -> str:
    name = spec.backend.lower()
    if "streaming" in name:
        return "streaming"
    if "lazy" in name:
        return "lazy"
    if "persistent" in name or "storage" in spec.execution_model:
        return "persistent"
    if name == "polars":
        return "eager"
    if spec.family == "dataframe":
        return "eager"
    if spec.family == "arrow":
        return "batch"
    return "default"


def _edge_relation(left: TargetSpec, right: TargetSpec) -> tuple[str, float]:
    left_module = left.adapter.rpartition(".")[0]
    right_module = right.adapter.rpartition(".")[0]
    if left_module and left_module == right_module and left.family == right.family:
        return "execution_mode_equivalence", 2.0
    if left.family == right.family:
        return "same_family_equivalence", 1.5
    return "cross_family_equivalence", 1.0
