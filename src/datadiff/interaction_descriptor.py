from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from datadiff.canonicalization import canonical_key, short_canonical_hash
from datadiff.ccs_ir import ContractCarryingRelationalIR, RelNodeIR


INTERACTION_DESCRIPTOR_SCHEMA_VERSION = "semantic-physical-interaction-v1"


@dataclass(frozen=True, slots=True)
class InteractionCoverageToken:
    category: str
    components: tuple[str, ...]

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INTERACTION_DESCRIPTOR_SCHEMA_VERSION,
            "category": self.category,
            "components": list(self.components),
        }

    @property
    def token_id(self) -> str:
        return "interaction-" + short_canonical_hash(self.identity_payload, 24)

    @property
    def label(self) -> str:
        return f"{self.category}:" + "|".join(self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "category": self.category,
            "components": list(self.components),
            "label": self.label,
        }


@dataclass(frozen=True, slots=True)
class InteractionDescriptor:
    semantic_digest: str
    layout_digest: str
    plan_fingerprints: tuple[str, ...]
    tokens: tuple[InteractionCoverageToken, ...]
    collision_count: int = 0

    @property
    def digest(self) -> str:
        return "interaction-descriptor-" + short_canonical_hash(
            {
                "schema_version": INTERACTION_DESCRIPTOR_SCHEMA_VERSION,
                "semantic_digest": self.semantic_digest,
                "layout_digest": self.layout_digest,
                "plan_fingerprints": list(self.plan_fingerprints),
                "tokens": [token.identity_payload for token in self.tokens],
            },
            64,
        )

    @property
    def coverage_tokens(self) -> tuple[str, ...]:
        return tuple(token.token_id for token in self.tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INTERACTION_DESCRIPTOR_SCHEMA_VERSION,
            "digest": self.digest,
            "semantic_digest": self.semantic_digest,
            "layout_digest": self.layout_digest,
            "plan_fingerprints": list(self.plan_fingerprints),
            "token_count": len(self.tokens),
            "collision_count": self.collision_count,
            "coverage_tokens": list(self.coverage_tokens),
            "tokens": [token.to_dict() for token in self.tokens],
        }

    def compact_dict(self) -> dict[str, Any]:
        """Return stable archive keys without repeating token explanations."""

        return {
            "schema_version": INTERACTION_DESCRIPTOR_SCHEMA_VERSION,
            "representation": "compact",
            "digest": self.digest,
            "semantic_digest": self.semantic_digest,
            "layout_digest": self.layout_digest,
            "plan_fingerprints": list(self.plan_fingerprints),
            "token_count": len(self.tokens),
            "collision_count": self.collision_count,
            "coverage_tokens": list(self.coverage_tokens),
        }


def build_interaction_descriptor(
    ir: ContractCarryingRelationalIR,
    raw_results: Mapping[str, Any],
) -> InteractionDescriptor:
    tokens: list[InteractionCoverageToken] = []
    node_states = [(node, _node_state(node)) for node in ir.nodes]

    for node, state in node_states:
        tokens.append(_token("semantic_state", state))
        for obligation in node.contract.test_obligations:
            tokens.append(
                _token(
                    "obligation_state",
                    (
                        f"obligation={obligation.obligation_id}",
                        f"kind={obligation.kind}",
                        *state,
                    ),
                )
            )
    for left, right in zip(ir.nodes, ir.nodes[1:]):
        tokens.append(
            _token(
                "semantic_transition",
                (
                    f"left={left.kind}",
                    f"right={right.kind}",
                    f"row={left.row_effect}->{right.row_effect}",
                    f"order={left.order_effect}->{right.order_effect}",
                    f"null={_null_state(left)}->{_null_state(right)}",
                ),
            )
        )
    for obligation in ir.program_obligations:
        tokens.append(
            _token(
                "program_obligation",
                (
                    f"obligation={obligation.obligation_id}",
                    f"scope={obligation.scope}",
                    *(f"fault={item}" for item in obligation.target_fault_models),
                ),
            )
        )

    for layout in ir.input_layouts:
        layout_state = (
            f"representation={layout.representation}",
            f"rows={_row_bucket(layout.row_count)}",
            f"chunks={_chunk_bucket(layout.chunk_count)}",
            f"dictionary={bool(layout.dictionary_columns)}",
        )
        tokens.append(_token("layout_state", layout_state))
        for node, state in node_states:
            tokens.append(
                _token(
                    "layout_semantic",
                    (*layout_state, f"op={node.kind}", *state[1:]),
                )
            )

    plan_fingerprints: list[str] = []
    for backend, raw in sorted(raw_results.items()):
        plan = raw.get("physical_plan") if isinstance(raw, Mapping) else None
        if not isinstance(plan, Mapping):
            continue
        backend_version = str(plan.get("backend_version", "") or "")
        mode = _backend_mode(str(backend))
        observations = _ok_observations(plan)
        for observation in observations:
            plan_kind = str(observation.get("plan_kind", "") or "")
            fingerprint = str(observation.get("fingerprint", "") or "")
            if fingerprint:
                plan_fingerprints.append(fingerprint)
            operators = tuple(
                str(item)
                for item in observation.get("operator_tokens", ()) or ()
                if str(item)
            )
            tokens.append(
                _token(
                    "plan_state",
                    (
                        f"backend={backend}",
                        f"mode={mode}",
                        f"version={backend_version}",
                        f"kind={plan_kind}",
                        f"operators={','.join(operators) or 'none'}",
                    ),
                )
            )
            for operator in sorted(set(operators)):
                for node, state in node_states:
                    tokens.append(
                        _token(
                            "semantic_physical",
                            (
                                f"backend={backend}",
                                f"mode={mode}",
                                f"plan_kind={plan_kind}",
                                f"operator={operator}",
                                f"op={node.kind}",
                                *state[1:],
                            ),
                        )
                    )

        for edge in _plan_edges(str(backend), observations):
            tokens.append(_token("plan_edge", edge))
            for node, state in node_states:
                tokens.append(
                    _token(
                        "plan_edge_semantic",
                        (*edge, f"op={node.kind}", *state[1:]),
                    )
                )
        sort_loss = _sort_loss_signal(observations)
        if sort_loss and any(_requires_order(node) for node in ir.nodes):
            tokens.append(
                _token(
                    "order_required_plan_sort_loss",
                    (
                        f"backend={backend}",
                        *sort_loss,
                        "semantic_order_required=true",
                        "offset_or_limit="
                        + str(any(node.kind in {"offset", "limit"} for node in ir.nodes)).lower(),
                    ),
                )
            )

    deduplicated, collision_count = _deduplicate_and_audit(tokens)
    return InteractionDescriptor(
        semantic_digest=ir.semantic_digest,
        layout_digest=ir.layout_digest,
        plan_fingerprints=tuple(sorted(set(plan_fingerprints))),
        tokens=deduplicated,
        collision_count=collision_count,
    )


def audit_interaction_descriptor_collisions(
    descriptors: Iterable[InteractionDescriptor],
) -> dict[str, Any]:
    resolved = tuple(descriptors)
    identities: dict[str, str] = {}
    collisions: list[dict[str, str]] = []
    token_count = 0
    for descriptor in resolved:
        for token in descriptor.tokens:
            token_count += 1
            identity = canonical_key(token.identity_payload)
            previous = identities.setdefault(token.token_id, identity)
            if previous != identity:
                collisions.append(
                    {
                        "token_id": token.token_id,
                        "left_identity": previous,
                        "right_identity": identity,
                    }
                )
    return {
        "schema_version": INTERACTION_DESCRIPTOR_SCHEMA_VERSION,
        "descriptor_count": len(resolved),
        "token_count": token_count,
        "unique_token_count": len(identities),
        "collision_count": len(collisions),
        "collisions": collisions,
    }


def _token(category: str, components: Iterable[str]) -> InteractionCoverageToken:
    return InteractionCoverageToken(
        category=str(category),
        components=tuple(str(item) for item in components if str(item)),
    )


def _node_state(node: RelNodeIR) -> tuple[str, ...]:
    logical_types = sorted(
        {
            column.logical_type
            for column in (*node.column_reads, *node.output_relation.columns)
        }
    )
    ordering = node.output_relation.ordering
    return (
        f"op={node.kind}",
        f"types={','.join(logical_types) or 'none'}",
        f"null={_null_state(node)}",
        f"row_effect={node.row_effect}",
        f"column_effect={node.column_effect}",
        f"order_effect={node.order_effect}",
        f"order_mode={ordering.mode}",
        f"order_observed={str(ordering.observed).lower()}",
        f"equivalence={node.contract.equivalence}",
    )


def _null_state(node: RelNodeIR) -> str:
    columns = tuple(node.output_relation.columns)
    if not columns:
        return "none"
    nullable = sum(bool(column.nullable) for column in columns)
    if nullable == 0:
        return "non_null"
    if nullable == len(columns):
        return "all_nullable"
    return "mixed_nullable"


def _ok_observations(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    observations = [
        item
        for item in plan.get("observations", ()) or ()
        if isinstance(item, Mapping) and item.get("status") == "ok"
    ]
    order = {"logical": 0, "optimized_logical": 1, "physical": 2}
    return sorted(
        observations,
        key=lambda item: (order.get(str(item.get("plan_kind", "")), 99), str(item.get("plan_kind", ""))),
    )


def _plan_edges(
    backend: str,
    observations: list[Mapping[str, Any]],
) -> list[tuple[str, ...]]:
    edges: list[tuple[str, ...]] = []
    for left, right in zip(observations, observations[1:]):
        left_counts = Counter(str(item) for item in left.get("operator_tokens", ()) or ())
        right_counts = Counter(str(item) for item in right.get("operator_tokens", ()) or ())
        added = tuple(sorted((right_counts - left_counts).elements()))
        removed = tuple(sorted((left_counts - right_counts).elements()))
        edges.append(
            (
                f"backend={backend}",
                f"edge={left.get('plan_kind', '')}->{right.get('plan_kind', '')}",
                f"changed={str(left.get('fingerprint', '') != right.get('fingerprint', '')).lower()}",
                f"added={','.join(added) or 'none'}",
                f"removed={','.join(removed) or 'none'}",
            )
        )
    return edges


def _sort_loss_signal(
    observations: list[Mapping[str, Any]],
) -> tuple[str, ...] | None:
    optimized = next(
        (item for item in observations if item.get("plan_kind") == "optimized_logical"),
        None,
    )
    physical = next(
        (item for item in observations if item.get("plan_kind") == "physical"),
        None,
    )
    if optimized is None or physical is None:
        return None
    logical_count = tuple(optimized.get("operator_tokens", ()) or ()).count("sort")
    physical_count = tuple(physical.get("operator_tokens", ()) or ()).count("sort")
    if physical_count >= logical_count:
        return None
    return (
        f"logical_sort_count={logical_count}",
        f"physical_sort_count={physical_count}",
    )


def _requires_order(node: RelNodeIR) -> bool:
    return bool(
        node.contract.order_observable
        or node.output_relation.ordering.observed
        or node.order_effect in {"observe", "define_and_observe"}
    )


def _deduplicate_and_audit(
    tokens: Iterable[InteractionCoverageToken],
) -> tuple[tuple[InteractionCoverageToken, ...], int]:
    by_id: dict[str, InteractionCoverageToken] = {}
    identities: dict[str, str] = {}
    collisions = 0
    for token in tokens:
        identity = canonical_key(token.identity_payload)
        previous = identities.setdefault(token.token_id, identity)
        if previous != identity:
            collisions += 1
            continue
        by_id[token.token_id] = token
    return tuple(by_id[key] for key in sorted(by_id)), collisions


def _row_bucket(row_count: int) -> str:
    value = max(0, int(row_count or 0))
    if value == 0:
        return "empty"
    if value == 1:
        return "singleton"
    if value <= 8:
        return "small"
    if value <= 64:
        return "medium"
    return "large"


def _chunk_bucket(chunk_count: int | None) -> str:
    if chunk_count is None:
        return "unknown"
    if chunk_count <= 1:
        return "single"
    if chunk_count <= 4:
        return "few"
    return "many"


def _backend_mode(backend: str) -> str:
    for suffix in ("streaming", "persistent", "lazy", "storage"):
        if suffix in backend:
            return suffix
    return "default"
