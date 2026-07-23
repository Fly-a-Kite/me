from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import random
from typing import Any, Iterable, Mapping, Sequence

from datadiff.backend_sampling import normalize_backend_pair
from datadiff.execution_lattice import (
    LatticeSelection,
    SemanticExecutionLattice,
    build_target_execution_lattice,
    select_budgeted_subgraph,
)
from datadiff.execution_cost_model import BackendCostModel


@dataclass(frozen=True, slots=True)
class LatticeBackendSelection:
    iteration: int
    mode: str
    active_backends: tuple[str, ...]
    omitted_backends: tuple[str, ...]
    priority_pairs: tuple[str, ...]
    coverage_after: dict[str, object]
    lattice_selection: dict[str, Any]

    @property
    def sampled(self) -> bool:
        return bool(self.omitted_backends)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy": "semantic_execution_lattice",
            "iteration": self.iteration,
            "mode": self.mode,
            "sampled": self.sampled,
            "active_backends": list(self.active_backends),
            "omitted_backends": list(self.omitted_backends),
            "priority_pairs": list(self.priority_pairs),
            "coverage_after": dict(self.coverage_after),
            "lattice_selection": dict(self.lattice_selection),
        }


class LatticeExecutionController:
    """Deterministic, budgeted backend-node selection for a fuzzing run."""

    def __init__(
        self,
        backends: Sequence[str],
        *,
        selector_mode: str,
        node_budget: float,
        versions: Mapping[str, str] | None = None,
        seed: int = 0,
        cost_model: BackendCostModel | None = None,
    ) -> None:
        self.backends = tuple(
            dict.fromkeys(str(name).strip() for name in backends if str(name).strip())
        )
        self.selector_mode = str(selector_mode)
        if self.selector_mode not in {"static", "shared_cost", "plan", "random"}:
            raise ValueError(f"unsupported lattice selector: {self.selector_mode}")
        minimum = 2.0 if len(self.backends) >= 2 else float(len(self.backends))
        self.node_budget = min(
            float(len(self.backends)),
            max(minimum, float(node_budget)),
        )
        self.lattice = build_target_execution_lattice(
            self.backends,
            versions=versions,
        )
        self.iteration = 0
        self.enabled = bool(len(self.backends) > int(self.node_budget))
        self.sample_size = min(len(self.backends), max(0, int(self.node_budget)))
        self.full_sweep_interval = 0
        self.calibration_cases = 0
        self.candidate_burst_cases = 0
        self._random = random.Random(int(seed))
        self.cost_model = cost_model
        self._mode_counts: Counter[str] = Counter()
        self._backend_counts: Counter[str] = Counter()
        self._pair_counts: Counter[str] = Counter()
        self._coverage_token_counts: Counter[str] = Counter()
        self._candidate_count = 0
        self._pairs = tuple(
            normalize_backend_pair(f"{left}|{right}")
            for left, right in combinations(self.backends, 2)
        )
        self._node_id_by_backend = {
            node.backend: node_id for node_id, node in self.lattice.nodes.items()
        }
        self._node_costs = {
            node_id: (
                self.cost_model.cost(node.backend)
                if self.cost_model is not None
                else 1.0
            )
            for node_id, node in self.lattice.nodes.items()
        }

    def select(
        self,
        *,
        priority_pairs: Iterable[str] = (),
        semantic_plan_fingerprint: Mapping[str, Any] | None = None,
        required_backends: Iterable[str] = (),
    ) -> LatticeBackendSelection:
        normalized_priority = tuple(
            dict.fromkeys(
                pair
                for pair in (normalize_backend_pair(item) for item in priority_pairs)
                if pair in self._pairs
            )
        )
        selection_iteration = self.iteration
        token_weights: dict[str, float] = {}
        normalized_required_backends = tuple(
            backend
            for backend in self.backends
            if backend in {
                str(item).strip()
                for item in required_backends
                if str(item).strip()
            }
        )
        if len(normalized_required_backends) >= 2:
            selected_node_ids = tuple(
                sorted(
                    self._node_id_by_backend[backend]
                    for backend in normalized_required_backends
                )
            )
            selected_set = set(selected_node_ids)
            selected_edges = tuple(
                edge
                for edge in self.lattice.edges
                if edge.source in selected_set and edge.target in selected_set
            )
            selected_edge_ids = tuple(edge.edge_id for edge in selected_edges)
            lattice_selection = LatticeSelection(
                selected_node_ids=selected_node_ids,
                selected_edge_ids=selected_edge_ids,
                covered_tokens=tuple(
                    sorted(
                        {
                            token
                            for edge in selected_edges
                            for token in edge.coverage_tokens
                        }
                    )
                ),
                total_cost=sum(
                    self._node_costs[node_id] for node_id in selected_node_ids
                ),
                budget=self.node_budget,
                omitted_edge_ids=tuple(
                    sorted(
                        edge.edge_id
                        for edge in self.lattice.edges
                        if edge.edge_id not in set(selected_edge_ids)
                    )
                ),
            )
            mode = "lattice_registered_family"
        elif not self.enabled:
            selected_node_ids = tuple(sorted(self.lattice.nodes))
            selected_edge_ids = tuple(edge.edge_id for edge in self.lattice.edges)
            lattice_selection = LatticeSelection(
                selected_node_ids=selected_node_ids,
                selected_edge_ids=selected_edge_ids,
                covered_tokens=tuple(
                    sorted({token for edge in self.lattice.edges for token in edge.coverage_tokens})
                ),
                total_cost=float(len(selected_node_ids)),
                budget=self.node_budget,
                omitted_edge_ids=(),
            )
            mode = "lattice_full_suite"
        elif self.selector_mode == "random":
            lattice_selection = self._random_selection()
            mode = "lattice_random"
        else:
            anchors = self._anchor_node_ids(normalized_priority)
            already_covered = (
                tuple(self._coverage_token_counts)
                if self.selector_mode in {"shared_cost", "plan"}
                else ()
            )
            operator_tokens = tuple(
                str(token)
                for token in (semantic_plan_fingerprint or {}).get("operator_tokens", [])
            )
            token_weights = self._token_weights(normalized_priority, operator_tokens)
            lattice_selection = select_budgeted_subgraph(
                self.lattice,
                budget=self.node_budget,
                node_costs=(
                    self._node_costs if self.selector_mode == "shared_cost" else None
                ),
                anchor_node_ids=anchors,
                already_covered_tokens=already_covered,
                token_weights=token_weights,
            )
            mode = f"lattice_{self.selector_mode}"
        active_set = {
            self.lattice.nodes[node_id].backend
            for node_id in lattice_selection.selected_node_ids
        }
        active = tuple(backend for backend in self.backends if backend in active_set)
        omitted = tuple(backend for backend in self.backends if backend not in active_set)
        self._record(active, lattice_selection)
        self._mode_counts[mode] += 1
        self.iteration += 1
        return LatticeBackendSelection(
            iteration=selection_iteration,
            mode=mode,
            active_backends=active,
            omitted_backends=omitted,
            priority_pairs=normalized_priority,
            coverage_after=self.coverage_summary(),
            lattice_selection={
                **lattice_selection.to_dict(),
                "selector_mode": self.selector_mode,
                "selected_backends": list(active),
                "omitted_backends": list(omitted),
                "semantic_plan_guidance": {
                    "fingerprint": dict(semantic_plan_fingerprint or {}),
                    "token_weights": dict(sorted(token_weights.items())),
                },
                "required_backends": list(normalized_required_backends),
                "cost_model": (
                    self.cost_model.to_dict()
                    if self.cost_model is not None
                    else {
                        "schema_version": "execution-cost-model-unit-fallback-v1",
                        "metric": "unit_node_cost",
                        "normalized_costs": {
                            backend: 1.0 for backend in self.backends
                        },
                    }
                ),
                "node_costs": {
                    self.lattice.nodes[node_id].backend: self._node_costs[node_id]
                    for node_id in sorted(self._node_costs)
                },
                "budget_satisfied": lattice_selection.total_cost <= self.node_budget,
            },
        )

    def record_outcome(
        self,
        *,
        candidate_detected: bool,
        candidate_families: Iterable[str] = (),
    ) -> dict[str, object]:
        if candidate_detected:
            self._candidate_count += 1
        return {
            "policy": "semantic_execution_lattice",
            "candidate_detected": bool(candidate_detected),
            "candidate_families": sorted(set(str(item) for item in candidate_families)),
            "candidate_count": self._candidate_count,
            "burst_armed": False,
            "burst_remaining": 0,
        }

    def coverage_summary(self) -> dict[str, object]:
        backend_values = [self._backend_counts[name] for name in self.backends]
        pair_values = [self._pair_counts[pair] for pair in self._pairs]
        return {
            "policy": "semantic_execution_lattice",
            "selector_mode": self.selector_mode,
            "node_budget": self.node_budget,
            "iterations": self.iteration,
            "mode_counts": dict(sorted(self._mode_counts.items())),
            "backend_min": min(backend_values, default=0),
            "backend_max": max(backend_values, default=0),
            "pair_min": min(pair_values, default=0),
            "pair_max": max(pair_values, default=0),
            "backend_counts": {name: self._backend_counts[name] for name in self.backends},
            "pair_counts": {pair: self._pair_counts[pair] for pair in self._pairs},
            "covered_token_count": len(self._coverage_token_counts),
            "coverage_token_counts": dict(sorted(self._coverage_token_counts.items())),
            "lattice_node_count": len(self.lattice.nodes),
            "lattice_edge_count": len(self.lattice.edges),
            "candidate_count": self._candidate_count,
        }

    def summary(self) -> dict[str, Any]:
        return {
            **self.coverage_summary(),
            "enabled": self.enabled,
            "lattice_schema_version": self.lattice.to_dict()["schema_version"],
            "cost_model": (
                self.cost_model.to_dict()
                if self.cost_model is not None
                else {"metric": "unit_node_cost", "digest": ""}
            ),
        }

    def _anchor_node_ids(self, priority_pairs: tuple[str, ...]) -> tuple[str, ...]:
        if self.selector_mode == "shared_cost" and self.backends:
            least_used = min(
                self.backends,
                key=lambda backend: (self._backend_counts[backend], backend),
            )
            node_id = self._node_id_by_backend.get(least_used)
            return (node_id,) if node_id else ()
        if self.selector_mode != "plan" or not priority_pairs:
            return ()
        anchors: list[str] = []
        for pair in priority_pairs:
            for backend in pair.split("|"):
                node_id = self._node_id_by_backend.get(backend)
                if node_id and node_id not in anchors:
                    anchors.append(node_id)
                if len(anchors) >= int(self.node_budget):
                    return tuple(anchors)
        return tuple(anchors)

    def _token_weights(
        self,
        priority_pairs: tuple[str, ...],
        operator_tokens: Iterable[str],
    ) -> dict[str, float]:
        if self.selector_mode != "plan":
            return {}
        weights = {f"backend_pair:{pair}": 4.0 for pair in priority_pairs}
        operators = {str(token).strip() for token in operator_tokens if str(token).strip()}
        if operators & {"hash_join", "nested_loop_join", "merge_join", "cross_join", "aggregate", "window"}:
            weights["relation:cross_family_equivalence"] = 3.0
        if operators & {"window", "sort", "topk", "limit"}:
            weights["relation:execution_mode_equivalence"] = 4.0
        if operators & {"filter", "projection", "distinct"}:
            weights["relation:same_family_equivalence"] = 2.0
        return weights

    def _random_selection(self) -> LatticeSelection:
        node_ids = sorted(self.lattice.nodes)
        selected = tuple(sorted(self._random.sample(node_ids, k=self.sample_size)))
        selected_set = set(selected)
        edges = [
            edge for edge in self.lattice.edges
            if edge.source in selected_set and edge.target in selected_set
        ]
        selected_edge_ids = tuple(edge.edge_id for edge in edges)
        covered_tokens = tuple(
            sorted({token for edge in edges for token in edge.coverage_tokens})
        )
        selected_edge_set = set(selected_edge_ids)
        return LatticeSelection(
            selected_node_ids=selected,
            selected_edge_ids=selected_edge_ids,
            covered_tokens=covered_tokens,
            total_cost=float(len(selected)),
            budget=self.node_budget,
            omitted_edge_ids=tuple(
                sorted(
                    edge.edge_id
                    for edge in self.lattice.edges
                    if edge.edge_id not in selected_edge_set
                )
            ),
        )

    def _record(self, active: tuple[str, ...], selection: LatticeSelection) -> None:
        self._backend_counts.update(active)
        self._pair_counts.update(
            normalize_backend_pair(f"{left}|{right}")
            for left, right in combinations(active, 2)
        )
        self._coverage_token_counts.update(selection.covered_tokens)
