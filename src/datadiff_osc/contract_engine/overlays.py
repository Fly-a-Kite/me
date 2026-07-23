from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.capability import SemanticPermission
from datadiff_osc.contract_engine.model import Endpoint


_AUTHORITY_TOKENS = (
    "family_id", "candidate_id", "root_id", "family:", "candidate:", "root:"
)


@dataclass(frozen=True, slots=True)
class BackendOverlay:
    overlay_id: str
    backend: str
    version_spec: str
    adapter_revision: str
    exact_preconditions: tuple[str, ...]
    affected_component: str
    permitted_relation: str
    evidence_source: str
    expiry_policy: str
    schema_version: str = "osc-backend-overlay-v1"

    def __post_init__(self) -> None:
        values = (
            self.overlay_id,
            self.backend,
            self.version_spec,
            self.adapter_revision,
            self.affected_component,
            self.permitted_relation,
            self.evidence_source,
            self.expiry_policy,
        )
        if not all(values):
            raise ValueError("backend overlay fields must be non-empty")
        authority_text = " ".join(
            (
                self.overlay_id,
                *self.exact_preconditions,
                self.affected_component,
                self.permitted_relation,
            )
        ).lower()
        if any(token in authority_text for token in _AUTHORITY_TOKENS):
            raise ValueError("backend overlay cannot contain family/root/candidate authority")
        if not self.exact_preconditions:
            raise ValueError("backend overlay requires exact preconditions")
        if len(self.exact_preconditions) != len(set(self.exact_preconditions)):
            raise ValueError("backend overlay exact preconditions must be unique")

    @property
    def digest(self) -> str:
        return stable_digest("osc-backend-overlay", self)

    def matches(self, endpoint: Endpoint, facts: frozenset[str]) -> bool:
        return (
            endpoint.backend == self.backend
            and endpoint.adapter_revision == self.adapter_revision
            and version_matches(endpoint.backend_version, self.version_spec)
            and set(self.exact_preconditions) <= set(facts)
        )

    def permission(self) -> SemanticPermission:
        return SemanticPermission(
            permission_id=self.overlay_id,
            affected_component=self.affected_component,
            relation_id=self.permitted_relation,
            exact_preconditions=self.exact_preconditions,
            overlay_digest=self.digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class OverlayRegistry:
    overlays: tuple[BackendOverlay, ...] = ()
    schema_version: str = "osc-overlay-registry-v1"

    def __post_init__(self) -> None:
        ids = [item.overlay_id for item in self.overlays]
        if len(ids) != len(set(ids)):
            raise ValueError("overlay IDs must be unique")

    @property
    def digest(self) -> str:
        return stable_digest("osc-overlay-registry", self)

    def resolve(
        self,
        endpoint: Endpoint,
        facts: frozenset[str],
        component: str,
        relation_id: str = "",
    ) -> tuple[SemanticPermission, ...]:
        return tuple(
            overlay.permission()
            for overlay in sorted(self.overlays, key=lambda item: item.overlay_id)
            if overlay.affected_component == component
            and (not relation_id or overlay.permitted_relation == relation_id)
            and overlay.matches(endpoint, facts)
        )

    def authorizes(
        self,
        endpoint: Endpoint,
        facts: frozenset[str],
        *,
        component: str,
        relation_id: str,
    ) -> bool:
        """Return only an exact component-and-relation scoped authorization."""

        return any(
            permission.permits(
                component=component,
                relation_id=relation_id,
                facts=facts,
            )
            for permission in self.resolve(
                endpoint, facts, component, relation_id=relation_id
            )
        )


def semantic_permission_errors(
    permissions: tuple[SemanticPermission, ...],
    *,
    components: tuple[str, ...],
    relation_id: str,
    facts: frozenset[str],
    overlay_digests: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate permissions without allowing one component to relax another."""

    errors: list[str] = []
    permission_ids = tuple(item.permission_id for item in permissions)
    if len(permission_ids) != len(set(permission_ids)):
        errors.append("duplicate_semantic_permission")
    allowed_components = set(components)
    allowed_overlays = set(overlay_digests)
    for permission in permissions:
        if not permission.valid:
            errors.append(f"invalid_semantic_permission:{permission.permission_id}")
            continue
        if permission.affected_component not in allowed_components:
            errors.append(
                f"permission_component_out_of_scope:{permission.permission_id}"
            )
        if permission.relation_id != relation_id:
            errors.append(
                f"permission_relation_out_of_scope:{permission.permission_id}"
            )
        if permission.overlay_digest not in allowed_overlays:
            errors.append(f"permission_overlay_unbound:{permission.permission_id}")
        if not set(permission.exact_preconditions) <= set(facts):
            errors.append(
                f"permission_precondition_missing:{permission.permission_id}"
            )
    return tuple(dict.fromkeys(errors))


def version_matches(version: str, spec: str) -> bool:
    if spec in {"", "*"}:
        return True
    current = _version_tuple(version)
    for clause in (item.strip() for item in spec.split(",") if item.strip()):
        match = re.fullmatch(r"(==|!=|>=|<=|>|<)?\s*([0-9][0-9A-Za-z._+-]*)", clause)
        if match is None:
            return False
        operator = match.group(1) or "=="
        expected = _version_tuple(match.group(2))
        comparison = (current > expected) - (current < expected)
        if operator == "==" and comparison != 0:
            return False
        if operator == "!=" and comparison == 0:
            return False
        if operator == ">=" and comparison < 0:
            return False
        if operator == "<=" and comparison > 0:
            return False
        if operator == ">" and comparison <= 0:
            return False
        if operator == "<" and comparison >= 0:
            return False
    return True


def _version_tuple(value: str) -> tuple[tuple[int, Any], ...]:
    tokens = re.findall(r"[0-9]+|[A-Za-z]+", str(value))
    return tuple((0, int(token)) if token.isdigit() else (1, token.lower()) for token in tokens)
