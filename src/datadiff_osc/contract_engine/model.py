from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datadiff_osc._canonical import (
    assert_deeply_immutable,
    frozen_pairs,
    immutable_value,
    stable_digest,
    to_primitive,
)
from datadiff_osc.schemas import ExecutionStatus, VerdictKind


CONTRACT_SCHEMA_VERSION = "osc-hypercontract-v1"
OBSERVATION_SCHEMA_VERSION = "osc-lossless-observation-v1"


def _validate_pairs(name: str, value: tuple[tuple[str, Any], ...]) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        for item in value
    ):
        raise ValueError(f"{name} must use canonical immutable string-key pairs")
    keys = tuple(item[0] for item in value)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} keys must be unique and canonically sorted")


@dataclass(frozen=True, slots=True)
class ComponentVerdict:
    component_id: str
    kind: VerdictKind
    reason: str = ""
    evidence: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component verdict identity must be non-empty")
        if not isinstance(self.kind, VerdictKind):
            raise ValueError("component verdict kind must use the frozen VerdictKind")
        _validate_pairs("component verdict evidence", self.evidence)
        assert_deeply_immutable(self.evidence)

    @classmethod
    def build(
        cls,
        component_id: str,
        kind: VerdictKind,
        reason: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> "ComponentVerdict":
        return cls(component_id, kind, reason, frozen_pairs(evidence))


@dataclass(frozen=True, slots=True)
class Verdict:
    kind: VerdictKind
    reason: str = ""
    components: tuple[ComponentVerdict, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, VerdictKind):
            raise ValueError("verdict kind must use the frozen VerdictKind")
        assert_deeply_immutable(self)

    @property
    def digest(self) -> str:
        return stable_digest("osc-verdict", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)

    @classmethod
    def aggregate(
        cls,
        components: tuple[ComponentVerdict, ...],
        reason: str = "",
    ) -> "Verdict":
        if not components:
            return cls(
                kind=VerdictKind.INCONCLUSIVE,
                reason=reason or "contract evaluation produced no component verdicts",
                components=(),
            )
        kinds = {item.kind for item in components}
        if VerdictKind.VIOLATED in kinds:
            kind = VerdictKind.VIOLATED
        elif VerdictKind.INCONCLUSIVE in kinds:
            kind = VerdictKind.INCONCLUSIVE
        elif VerdictKind.INAPPLICABLE in kinds:
            kind = VerdictKind.INAPPLICABLE
        else:
            kind = VerdictKind.SATISFIED
        return cls(kind=kind, reason=reason, components=components)


@dataclass(frozen=True, slots=True)
class Endpoint:
    endpoint_id: str
    case_digest: str
    backend: str
    backend_version: str
    adapter_revision: str
    execution_mode: str
    physical_layout: str
    optimizer_config: tuple[tuple[str, Any], ...] = ()
    capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        required = (
            self.endpoint_id,
            self.case_digest,
            self.backend,
            self.backend_version,
            self.adapter_revision,
            self.execution_mode,
            self.physical_layout,
        )
        if not all(str(value) for value in required):
            raise ValueError("endpoint identity and scope fields must be non-empty")
        if any(not isinstance(value, str) or not value for value in self.capabilities):
            raise ValueError("endpoint capabilities must be non-empty strings")
        _validate_pairs("endpoint optimizer configuration", self.optimizer_config)
        assert_deeply_immutable(self.optimizer_config)
        assert_deeply_immutable(self)

    @classmethod
    def build(
        cls,
        *,
        endpoint_id: str,
        case_digest: str,
        backend: str,
        backend_version: str,
        adapter_revision: str,
        execution_mode: str,
        physical_layout: str,
        optimizer_config: dict[str, Any] | None = None,
        capabilities: set[str] | frozenset[str] = frozenset(),
    ) -> "Endpoint":
        return cls(
            endpoint_id=endpoint_id,
            case_digest=case_digest,
            backend=backend,
            backend_version=backend_version,
            adapter_revision=adapter_revision,
            execution_mode=execution_mode,
            physical_layout=physical_layout,
            optimizer_config=frozen_pairs(optimizer_config),
            capabilities=frozenset(str(item) for item in capabilities),
        )

    @property
    def digest(self) -> str:
        return stable_digest("osc-endpoint", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class SchemaField:
    name: str
    logical_type: str
    nullable: bool

    def __post_init__(self) -> None:
        if not self.name or not self.logical_type or not isinstance(self.nullable, bool):
            raise ValueError("schema field name and logical type must be non-empty")


@dataclass(frozen=True, slots=True)
class Observation:
    endpoint_id: str
    status: str
    schema: tuple[SchemaField, ...] = ()
    rows: tuple[tuple[Any, ...], ...] = ()
    error_category: str = ""
    error_type: str = ""
    error_message: str = ""
    execution_metadata: tuple[tuple[str, Any], ...] = ()
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.endpoint_id:
            raise ValueError("observation endpoint_id must be non-empty")
        if self.status not in {item.value for item in ExecutionStatus}:
            raise ValueError(f"unknown observation status: {self.status}")
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unsupported observation schema version")
        if self.status == ExecutionStatus.OK.value and any(
            (self.error_category, self.error_type, self.error_message)
        ):
            raise ValueError("OK observation cannot carry error fields")
        if (
            self.status == ExecutionStatus.SEMANTIC_ERROR.value
            and not self.error_category
        ):
            raise ValueError("semantic-error observation requires an error category")
        if self.status != ExecutionStatus.OK.value and self.rows:
            raise ValueError("non-OK observation cannot carry result rows")
        if self.schema and any(len(row) != len(self.schema) for row in self.rows):
            raise ValueError("observation row width must match the declared schema")
        try:
            _validate_pairs("observation execution metadata", self.execution_metadata)
            assert_deeply_immutable(self.schema)
            assert_deeply_immutable(self.rows)
            assert_deeply_immutable(self.execution_metadata)
            assert_deeply_immutable(self)
        except TypeError as exc:
            raise ValueError(
                "observation payload must be deeply immutable; use Observation.build"
            ) from exc
        if self.rows and not self.schema:
            raise ValueError("result rows require an explicit lossless schema")

    @classmethod
    def build(
        cls,
        *,
        endpoint_id: str,
        status: str,
        schema: tuple[SchemaField, ...] | list[SchemaField] = (),
        rows: tuple[tuple[Any, ...], ...] | list[list[Any]] = (),
        error_category: str = "",
        error_type: str = "",
        error_message: str = "",
        execution_metadata: dict[str, Any] | None = None,
    ) -> "Observation":
        return cls(
            endpoint_id=endpoint_id,
            status=status,
            schema=tuple(schema),
            rows=tuple(
                tuple(immutable_value(value) for value in row)
                for row in rows
            ),
            error_category=error_category,
            error_type=error_type,
            error_message=error_message,
            execution_metadata=frozen_pairs(execution_metadata),
        )

    @property
    def cardinality(self) -> int:
        return len(self.rows)

    @property
    def digest(self) -> str:
        return stable_digest("osc-observation", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class EndpointRequirement:
    endpoint_id: str
    backend: str = "*"
    version_spec: str = "*"
    execution_modes: frozenset[str] = frozenset()
    physical_layouts: frozenset[str] = frozenset()
    required_capabilities: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.endpoint_id or not self.backend or not self.version_spec:
            raise ValueError("endpoint requirement identity and scope must be non-empty")
        for name, values in (
            ("execution modes", self.execution_modes),
            ("physical layouts", self.physical_layouts),
            ("required capabilities", self.required_capabilities),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"endpoint requirement {name} must be strings")
        assert_deeply_immutable(self)


@dataclass(frozen=True, slots=True)
class RelationObligation:
    obligation_id: str
    relation_id: str
    endpoint_ids: tuple[str, ...]
    components: tuple[str, ...]
    parameters: tuple[tuple[str, Any], ...] = ()
    strength: str = ""
    relation_properties: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.obligation_id or not self.relation_id:
            raise ValueError("relation obligation identity must be non-empty")
        if not self.endpoint_ids:
            raise ValueError("relation obligation needs at least one endpoint")
        if len(set(self.endpoint_ids)) != len(self.endpoint_ids):
            raise ValueError("relation obligation endpoint IDs must be unique")
        if not self.components or len(set(self.components)) != len(self.components):
            raise ValueError("relation obligation components must be non-empty and unique")
        if any(not item for item in self.components):
            raise ValueError("relation obligation component identity must be non-empty")
        _validate_pairs("relation parameters", self.parameters)
        assert_deeply_immutable(self.parameters)
        assert_deeply_immutable(self)

    @classmethod
    def build(
        cls,
        *,
        obligation_id: str,
        relation_id: str,
        endpoint_ids: tuple[str, ...],
        components: tuple[str, ...],
        parameters: dict[str, Any] | None = None,
        strength: str = "",
        relation_properties: set[str] | frozenset[str] = frozenset(),
    ) -> "RelationObligation":
        return cls(
            obligation_id=obligation_id,
            relation_id=relation_id,
            endpoint_ids=tuple(endpoint_ids),
            components=tuple(components),
            parameters=frozen_pairs(parameters),
            strength=strength,
            relation_properties=frozenset(relation_properties),
        )


@dataclass(frozen=True, slots=True)
class HyperContract:
    contract_id: str
    endpoint_requirements: tuple[EndpointRequirement, ...]
    preconditions: tuple[str, ...]
    observations: tuple[str, ...]
    obligations: tuple[RelationObligation, ...]
    strength: str
    derivation_digest: str
    registry_digest: str
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("unsupported HyperContract schema version")
        required_text = (
            self.contract_id,
            self.strength,
            self.derivation_digest,
            self.registry_digest,
        )
        if not all(required_text):
            raise ValueError(
                "contract identity, strength, derivation and registry digests must be non-empty"
            )
        endpoint_ids = [item.endpoint_id for item in self.endpoint_requirements]
        if not endpoint_ids or len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError("contract endpoint requirements must be non-empty and unique")
        if not self.observations or len(set(self.observations)) != len(self.observations):
            raise ValueError("contract observations must be non-empty and unique")
        if not self.obligations:
            raise ValueError("contract requires at least one relation obligation")
        obligation_ids = [item.obligation_id for item in self.obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise ValueError("contract obligation IDs must be unique")
        if len(self.preconditions) != len(set(self.preconditions)):
            raise ValueError("contract preconditions must be unique")
        known = set(endpoint_ids)
        for obligation in self.obligations:
            unknown = set(obligation.endpoint_ids) - known
            if unknown:
                raise ValueError(
                    f"obligation {obligation.obligation_id} references unknown endpoints: {sorted(unknown)}"
                )
        assert_deeply_immutable(self)

    @property
    def digest(self) -> str:
        return stable_digest("osc-hypercontract", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)
