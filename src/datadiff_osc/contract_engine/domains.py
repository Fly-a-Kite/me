from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from datadiff_osc._canonical import frozen_pairs, stable_digest, to_primitive


class SchemaKnowledge(str, Enum):
    KNOWN = "known"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class NullDomain(str, Enum):
    NONE = "none"
    MAYBE = "maybe"
    PRESENT = "present"
    UNKNOWN = "unknown"


class SpecialFloatDomain(str, Enum):
    NONE = "none"
    MAYBE = "maybe"
    PRESENT = "present"
    UNKNOWN = "unknown"


class CardinalityDomain(str, Enum):
    EMPTY = "empty"
    SINGLETON = "singleton"
    MULTIPLE = "multiple"
    NONEMPTY = "nonempty"
    UNKNOWN = "unknown"


class MultiplicityDomain(str, Enum):
    UNIQUE = "unique"
    MAY_DUPLICATE = "may_duplicate"
    SET_OUTPUT = "set_output"
    UNKNOWN = "unknown"


class OrderDomain(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    TOTAL = "total"
    UNKNOWN = "unknown"


class TieDeterminismDomain(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    DETERMINISTIC = "deterministic"
    NONDETERMINISTIC = "nondeterministic"
    UNKNOWN = "unknown"


class PartitionOrderDomain(str, Enum):
    UNPARTITIONED = "unpartitioned"
    ORDERED = "ordered"
    UNORDERED = "unordered"
    UNKNOWN = "unknown"


class LayoutDomain(str, Enum):
    LOGICAL_ROWS = "logical_rows"
    CONTIGUOUS = "contiguous"
    SLICED = "sliced"
    CHUNKED = "chunked"
    DICTIONARY = "dictionary"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class DeterminismDomain(str, Enum):
    DETERMINISTIC = "deterministic"
    NONDETERMINISTIC = "nondeterministic"
    UNKNOWN = "unknown"


class DefinednessDomain(str, Enum):
    DEFINED = "defined"
    SEMANTIC_DOMAIN_ERROR = "semantic_domain_error"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


_JOIN_TABLES: dict[type[Enum], dict[frozenset[Enum], Enum]] = {
    NullDomain: {
        frozenset({NullDomain.NONE, NullDomain.PRESENT}): NullDomain.MAYBE,
        frozenset({NullDomain.NONE, NullDomain.MAYBE}): NullDomain.MAYBE,
        frozenset({NullDomain.PRESENT, NullDomain.MAYBE}): NullDomain.MAYBE,
    },
    SpecialFloatDomain: {
        frozenset({SpecialFloatDomain.NONE, SpecialFloatDomain.PRESENT}): SpecialFloatDomain.MAYBE,
        frozenset({SpecialFloatDomain.NONE, SpecialFloatDomain.MAYBE}): SpecialFloatDomain.MAYBE,
        frozenset({SpecialFloatDomain.PRESENT, SpecialFloatDomain.MAYBE}): SpecialFloatDomain.MAYBE,
    },
    CardinalityDomain: {
        frozenset({CardinalityDomain.SINGLETON, CardinalityDomain.MULTIPLE}): CardinalityDomain.NONEMPTY,
        frozenset({CardinalityDomain.SINGLETON, CardinalityDomain.NONEMPTY}): CardinalityDomain.NONEMPTY,
        frozenset({CardinalityDomain.MULTIPLE, CardinalityDomain.NONEMPTY}): CardinalityDomain.NONEMPTY,
    },
    OrderDomain: {
        frozenset({OrderDomain.PARTIAL, OrderDomain.TOTAL}): OrderDomain.PARTIAL,
        frozenset({OrderDomain.NONE, OrderDomain.PARTIAL}): OrderDomain.UNKNOWN,
        frozenset({OrderDomain.NONE, OrderDomain.TOTAL}): OrderDomain.UNKNOWN,
    },
}


def finite_join(left: Enum, right: Enum) -> Enum:
    if type(left) is not type(right):
        raise TypeError("finite domain join requires identical enum types")
    if left == right:
        return left
    enum_type = type(left)
    unknown = next((item for item in enum_type if item.value == "unknown"), None)
    if left == unknown or right == unknown:
        return unknown  # type: ignore[return-value]
    return _JOIN_TABLES.get(enum_type, {}).get(frozenset({left, right}), unknown)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class AbstractState:
    schema_knowledge: SchemaKnowledge = SchemaKnowledge.UNKNOWN
    schema: tuple[tuple[str, str, bool], ...] = ()
    nulls: NullDomain = NullDomain.UNKNOWN
    special_floats: SpecialFloatDomain = SpecialFloatDomain.UNKNOWN
    cardinality: CardinalityDomain = CardinalityDomain.UNKNOWN
    multiplicity: MultiplicityDomain = MultiplicityDomain.UNKNOWN
    evaluation_order: OrderDomain = OrderDomain.UNKNOWN
    presentation_order: OrderDomain = OrderDomain.UNKNOWN
    tie_determinism: TieDeterminismDomain = TieDeterminismDomain.UNKNOWN
    partition_order: PartitionOrderDomain = PartitionOrderDomain.UNKNOWN
    partition_keys: tuple[str, ...] = ()
    layout: LayoutDomain = LayoutDomain.UNKNOWN
    determinism: DeterminismDomain = DeterminismDomain.UNKNOWN
    definedness: DefinednessDomain = DefinednessDomain.UNKNOWN
    facts: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def initial(
        cls,
        *,
        schema: tuple[tuple[str, str, bool], ...] = (),
        has_nulls: bool | None = None,
        has_special_floats: bool | None = None,
        row_count: int | None = None,
        layout: LayoutDomain = LayoutDomain.LOGICAL_ROWS,
    ) -> "AbstractState":
        if row_count is None:
            cardinality = CardinalityDomain.UNKNOWN
        elif row_count == 0:
            cardinality = CardinalityDomain.EMPTY
        elif row_count == 1:
            cardinality = CardinalityDomain.SINGLETON
        else:
            cardinality = CardinalityDomain.MULTIPLE
        return cls(
            schema_knowledge=SchemaKnowledge.KNOWN if schema else SchemaKnowledge.UNKNOWN,
            schema=tuple(schema),
            nulls=(NullDomain.PRESENT if has_nulls else NullDomain.NONE) if has_nulls is not None else NullDomain.UNKNOWN,
            special_floats=(SpecialFloatDomain.PRESENT if has_special_floats else SpecialFloatDomain.NONE) if has_special_floats is not None else SpecialFloatDomain.UNKNOWN,
            cardinality=cardinality,
            multiplicity=MultiplicityDomain.MAY_DUPLICATE,
            evaluation_order=OrderDomain.NONE,
            presentation_order=OrderDomain.NONE,
            tie_determinism=TieDeterminismDomain.NOT_APPLICABLE,
            partition_order=PartitionOrderDomain.UNPARTITIONED,
            layout=layout,
            determinism=DeterminismDomain.DETERMINISTIC,
            definedness=DefinednessDomain.DEFINED,
        )

    def evolve(self, **changes: Any) -> "AbstractState":
        if "facts" in changes and isinstance(changes["facts"], dict):
            changes["facts"] = frozen_pairs(changes["facts"])
        return replace(self, **changes)

    @property
    def digest(self) -> str:
        return stable_digest("osc-abstract-state", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


@dataclass(frozen=True, slots=True)
class ObservationDemand:
    status: bool = True
    schema_names: bool = True
    schema_types: bool = False
    nullability: bool = False
    cardinality: bool = True
    row_membership: bool = True
    duplicate_multiplicity: bool = True
    presentation_order: bool = False
    evaluation_order: bool = False
    partition_order: bool = False
    tie_determinism: bool = False
    numeric: bool = False
    null_semantics: bool = False
    special_float_roles: tuple[str, ...] = ()
    error_category: bool = False
    layout: bool = False
    determinism: bool = False
    demanded_columns: frozenset[str] = frozenset()

    def evolve(self, **changes: Any) -> "ObservationDemand":
        return replace(self, **changes)

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                "status", "schema_names", "schema_types", "nullability",
                "cardinality", "row_membership", "duplicate_multiplicity",
                "presentation_order", "evaluation_order", "partition_order",
                "tie_determinism", "numeric", "null_semantics",
                "error_category", "layout", "determinism",
            )
            if bool(getattr(self, name))
        )

