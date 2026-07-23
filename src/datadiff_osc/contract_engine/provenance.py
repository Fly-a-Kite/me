from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.model import Endpoint


@dataclass(frozen=True, slots=True)
class ImplementationLineage:
    endpoint_id: str
    frontend: str
    logical_engine: str
    physical_kernel: str
    adapter_lowering: str
    shared_dependencies: frozenset[str] = frozenset()
    schema_version: str = "osc-implementation-lineage-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-implementation-lineage", self)

    @property
    def independence_key(self) -> tuple[str, str, str, str]:
        return (
            self.frontend,
            self.logical_engine,
            self.physical_kernel,
            self.adapter_lowering,
        )


@dataclass(frozen=True, slots=True)
class EndpointCoverCandidate:
    endpoint: Endpoint
    covers: frozenset[str]
    predicted_cost: float
    lineage: ImplementationLineage


@dataclass(frozen=True, slots=True)
class EndpointCoverPlan:
    endpoint_ids: tuple[str, ...]
    covered_obligations: frozenset[str]
    uncovered_obligations: frozenset[str]
    independent_lineage_count: int
    total_predicted_cost: float
    schema_version: str = "osc-endpoint-cover-plan-v1"

    @property
    def digest(self) -> str:
        return stable_digest("osc-endpoint-cover-plan", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def select_endpoint_cover(
    candidates: tuple[EndpointCoverCandidate, ...],
    required_obligations: frozenset[str],
    *,
    required_endpoint_ids: frozenset[str] = frozenset(),
) -> EndpointCoverPlan:
    """Deterministic greedy cover; lineage affects selection, never verdict."""

    selected: list[EndpointCoverCandidate] = []
    covered: set[str] = set()
    lineage_keys: set[tuple[str, str, str, str]] = set()
    remaining = list(candidates)
    while set(required_endpoint_ids) - {item.endpoint.endpoint_id for item in selected} or set(required_obligations) - covered:
        best: EndpointCoverCandidate | None = None
        best_key: tuple[float, int, float, str] | None = None
        for item in remaining:
            mandatory = int(item.endpoint.endpoint_id in required_endpoint_ids)
            new_cover = len(set(item.covers) & (set(required_obligations) - covered))
            lineage_bonus = int(item.lineage.independence_key not in lineage_keys)
            cost = max(float(item.predicted_cost), 1e-9)
            key = (
                mandatory * 1_000_000.0 + (new_cover + 0.25 * lineage_bonus) / cost,
                lineage_bonus,
                -cost,
                item.endpoint.endpoint_id,
            )
            if best_key is None or key > best_key:
                best, best_key = item, key
        if best is None or (
            best.endpoint.endpoint_id not in required_endpoint_ids
            and not (set(best.covers) & (set(required_obligations) - covered))
        ):
            break
        selected.append(best)
        remaining.remove(best)
        covered.update(best.covers)
        lineage_keys.add(best.lineage.independence_key)
    return EndpointCoverPlan(
        endpoint_ids=tuple(item.endpoint.endpoint_id for item in selected),
        covered_obligations=frozenset(covered & set(required_obligations)),
        uncovered_obligations=frozenset(set(required_obligations) - covered),
        independent_lineage_count=len(lineage_keys),
        total_predicted_cost=sum(item.predicted_cost for item in selected),
    )

