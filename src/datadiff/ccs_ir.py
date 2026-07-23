from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from datadiff.canonicalization import canonical_key, short_canonical_hash
from datadiff.dsl import (
    Case,
    Operation,
    Program,
    SortKey,
    TableData,
    coerce_operation,
    normalize_sort_keys,
)
from datadiff.operation_semantics import (
    aggregate_alias,
    aggregate_column,
    aggregate_func,
    aggregate_specs,
    coalesce_sources,
    condition_column,
    condition_cmp,
    expr_kind,
    expr_numerator,
    expr_other,
    expr_payload,
    expr_source,
    groupby_keys,
    join_left_keys,
    join_right_keys,
    normalized_order_by_keys,
    op_column,
    op_columns,
    op_kind,
    op_output_alias,
    op_partition_columns,
    op_right_columns,
    op_table,
)
from datadiff.program_state import ProgramState, apply_operation_state
from datadiff.program_obligations import (
    ProgramObligationIR,
    infer_program_obligations,
)
from datadiff.util import unique_preserve_order


CCS_IR_SCHEMA_VERSION = "contract-carrying-semantic-relational-ir-v5"


@dataclass(frozen=True, slots=True)
class CanonicalPayload:
    """Immutable, deterministic mapping used by CCS-IR extension attributes."""

    canonical: str
    _value: dict[str, Any] = field(repr=False, compare=False)

    @classmethod
    def build(cls, value: Mapping[str, Any] | None = None) -> "CanonicalPayload":
        payload = copy.deepcopy(dict(value or {}))
        return cls(canonical=canonical_key(payload), _value=payload)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._value)


@dataclass(frozen=True, slots=True)
class ColumnIR:
    column_id: str
    name: str
    logical_type: str
    nullable: bool
    lineage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_id": self.column_id,
            "name": self.name,
            "logical_type": self.logical_type,
            "nullable": self.nullable,
            "lineage": list(self.lineage),
        }


@dataclass(frozen=True, slots=True)
class ColumnRefIR:
    relation_id: str
    column_id: str
    name: str
    logical_type: str
    nullable: bool

    @classmethod
    def from_column(cls, relation_id: str, column: ColumnIR) -> "ColumnRefIR":
        return cls(
            relation_id=relation_id,
            column_id=column.column_id,
            name=column.name,
            logical_type=column.logical_type,
            nullable=column.nullable,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "column_id": self.column_id,
            "name": self.name,
            "logical_type": self.logical_type,
            "nullable": self.nullable,
        }


@dataclass(frozen=True, slots=True)
class OrderKeyIR:
    column_id: str
    column_name: str
    ascending: bool
    nulls: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column_id": self.column_id,
            "column_name": self.column_name,
            "ascending": self.ascending,
            "nulls": self.nulls,
        }


@dataclass(frozen=True, slots=True)
class OrderingIR:
    mode: str = "none"
    observed: bool = False
    keys: tuple[OrderKeyIR, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "observed": self.observed,
            "keys": [key.to_dict() for key in self.keys],
        }


@dataclass(frozen=True, slots=True)
class RelationIR:
    relation_id: str
    name: str
    columns: tuple[ColumnIR, ...]
    ordering: OrderingIR = OrderingIR()

    @property
    def schema_digest(self) -> str:
        return f"schema-{short_canonical_hash([column.to_dict() for column in self.columns], 24)}"

    def column(self, name: str) -> ColumnIR | None:
        return next((column for column in self.columns if column.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "name": self.name,
            "schema_digest": self.schema_digest,
            "columns": [column.to_dict() for column in self.columns],
            "ordering": self.ordering.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class InputLayoutIR:
    relation_id: str
    table_name: str
    representation: str
    row_count: int
    chunk_count: int | None = None
    dictionary_columns: tuple[str, ...] = ()
    attributes: CanonicalPayload = field(default_factory=CanonicalPayload.build)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "table_name": self.table_name,
            "representation": self.representation,
            "row_count": self.row_count,
            "chunk_count": self.chunk_count,
            "dictionary_columns": list(self.dictionary_columns),
            "attributes": self.attributes.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OperationIR:
    """Canonical backend-neutral operation that lowers without a legacy escape hatch."""

    kind: str
    arguments: CanonicalPayload

    @classmethod
    def from_operation(cls, operation: Mapping[str, Any]) -> "OperationIR":
        payload = copy.deepcopy(dict(operation))
        kind = op_kind(payload, default="unknown")
        payload.pop("op", None)
        payload.pop("kind", None)
        return cls(
            kind=kind,
            arguments=CanonicalPayload.build(_canonical_operation_arguments(kind, payload)),
        )

    def to_operation(self) -> Operation:
        return coerce_operation({"op": self.kind, **self.arguments.to_dict()})

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "arguments": self.arguments.to_dict()}


@dataclass(frozen=True, slots=True)
class ScalarExpressionIR:
    expression_id: str
    kind: str
    inputs: tuple[ColumnRefIR, ...]
    result_type: str
    nullable: bool
    attributes: CanonicalPayload

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression_id": self.expression_id,
            "kind": self.kind,
            "inputs": [column.to_dict() for column in self.inputs],
            "result_type": self.result_type,
            "nullable": self.nullable,
            "attributes": self.attributes.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AggregateCallIR:
    aggregate_id: str
    function: str
    input_column: ColumnRefIR | None
    output_column_id: str
    output_name: str
    result_type: str
    nullable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "function": self.function,
            "input_column": self.input_column.to_dict() if self.input_column else None,
            "output_column_id": self.output_column_id,
            "output_name": self.output_name,
            "result_type": self.result_type,
            "nullable": self.nullable,
        }


@dataclass(frozen=True, slots=True)
class TestObligationIR:
    obligation_id: str
    kind: str
    transformation: str
    expected_relation: str
    preconditions: tuple[str, ...]
    oracle: str
    target_fault_models: tuple[str, ...] = ()
    estimated_cost: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "kind": self.kind,
            "transformation": self.transformation,
            "expected_relation": self.expected_relation,
            "preconditions": list(self.preconditions),
            "oracle": self.oracle,
            "target_fault_models": list(self.target_fault_models),
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True, slots=True)
class SemanticContractIR:
    equivalence: str
    definedness: str
    order_observable: bool
    duplicate_sensitive: bool
    null_sensitive: bool
    numeric_policy: str
    error_policy: str
    preconditions: tuple[str, ...] = ()
    postconditions: tuple[str, ...] = ()
    test_obligations: tuple[TestObligationIR, ...] = ()

    @property
    def metamorphic_relations(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation_id
            for obligation in self.test_obligations
            if obligation.kind == "metamorphic_relation"
        )

    @property
    def fault_models(self) -> tuple[str, ...]:
        return tuple(
            obligation.obligation_id
            for obligation in self.test_obligations
            if obligation.kind == "fault_model"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "equivalence": self.equivalence,
            "definedness": self.definedness,
            "order_observable": self.order_observable,
            "duplicate_sensitive": self.duplicate_sensitive,
            "null_sensitive": self.null_sensitive,
            "numeric_policy": self.numeric_policy,
            "error_policy": self.error_policy,
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "test_obligations": [
                obligation.to_dict() for obligation in self.test_obligations
            ],
        }


@dataclass(frozen=True, slots=True)
class RelNodeIR:
    node_id: str
    index: int
    kind: str
    input_relations: tuple[str, ...]
    output_relation: RelationIR
    column_reads: tuple[ColumnRefIR, ...]
    scalar_expressions: tuple[ScalarExpressionIR, ...]
    aggregates: tuple[AggregateCallIR, ...]
    required_capabilities: tuple[str, ...]
    contract: SemanticContractIR
    row_effect: str
    column_effect: str
    order_effect: str
    operation: OperationIR

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "index": self.index,
            "kind": self.kind,
            "input_relations": list(self.input_relations),
            "output_relation": self.output_relation.to_dict(),
            "column_reads": [column.to_dict() for column in self.column_reads],
            "scalar_expressions": [expression.to_dict() for expression in self.scalar_expressions],
            "aggregates": [aggregate.to_dict() for aggregate in self.aggregates],
            "required_capabilities": list(self.required_capabilities),
            "contract": self.contract.to_dict(),
            "row_effect": self.row_effect,
            "column_effect": self.column_effect,
            "order_effect": self.order_effect,
            "operation": self.operation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ContractCarryingRelationalIR:
    program_id: str
    seed: int
    primary_relation_id: str
    source_relations: tuple[RelationIR, ...]
    input_layouts: tuple[InputLayoutIR, ...]
    nodes: tuple[RelNodeIR, ...]
    program_obligations: tuple[ProgramObligationIR, ...]
    output_relation_id: str
    required_capabilities: tuple[str, ...]
    testing_objectives: tuple[str, ...]
    schema_version: str = CCS_IR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def digest(self) -> str:
        return f"ccs-ir-{short_canonical_hash({'semantic': self.semantic_digest, 'layout': self.layout_digest}, 64)}"

    @property
    def semantic_digest(self) -> str:
        return f"ccs-semantic-{short_canonical_hash(self.semantic_payload(), 64)}"

    @property
    def layout_digest(self) -> str:
        return f"ccs-layout-{short_canonical_hash(self.layout_payload(), 64)}"

    @property
    def syntax_digest(self) -> str:
        return f"ccs-syntax-{short_canonical_hash({'operations': [node.operation.to_dict() for node in self.nodes]}, 64)}"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "primary_relation_id": self.primary_relation_id,
            "source_relations": [relation.to_dict() for relation in self.source_relations],
            "nodes": [node.to_dict() for node in self.nodes],
            "program_obligations": [
                obligation.to_dict() for obligation in self.program_obligations
            ],
            "output_relation_id": self.output_relation_id,
            "required_capabilities": list(self.required_capabilities),
            "testing_objectives": list(self.testing_objectives),
        }

    def layout_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_layouts": [layout.to_dict() for layout in self.input_layouts],
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "semantic_digest": self.semantic_digest,
            "layout_digest": self.layout_digest,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "digest": self.digest,
            "semantic_digest": self.semantic_digest,
            "layout_digest": self.layout_digest,
            "syntax_digest": self.syntax_digest,
            "node_count": len(self.nodes),
            "source_relation_count": len(self.source_relations),
            "program_obligation_count": len(self.program_obligations),
            "required_capabilities": list(self.required_capabilities),
            "testing_objectives": list(self.testing_objectives),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "program_id": self.program_id,
            "seed": self.seed,
            "primary_relation_id": self.primary_relation_id,
            "source_relations": [relation.to_dict() for relation in self.source_relations],
            "input_layouts": [layout.to_dict() for layout in self.input_layouts],
            "nodes": [node.to_dict() for node in self.nodes],
            "program_obligations": [
                obligation.to_dict() for obligation in self.program_obligations
            ],
            "output_relation_id": self.output_relation_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContractCarryingRelationalIR":
        ir = cls(
            program_id=str(data.get("program_id", "")),
            seed=int(data.get("seed", 0)),
            primary_relation_id=str(data.get("primary_relation_id", "")),
            source_relations=tuple(
                _relation_from_dict(item)
                for item in data.get("source_relations", ())
                if isinstance(item, Mapping)
            ),
            input_layouts=tuple(
                _input_layout_from_dict(item)
                for item in data.get("input_layouts", ())
                if isinstance(item, Mapping)
            ),
            nodes=tuple(
                _node_from_dict(item)
                for item in data.get("nodes", ())
                if isinstance(item, Mapping)
            ),
            program_obligations=tuple(
                ProgramObligationIR.from_dict(item)
                for item in data.get("program_obligations", ())
                if isinstance(item, Mapping)
            ),
            output_relation_id=str(data.get("output_relation_id", "")),
            required_capabilities=tuple(
                str(item) for item in data.get("required_capabilities", ())
            ),
            testing_objectives=tuple(
                str(item) for item in data.get("testing_objectives", ())
            ),
            schema_version=str(data.get("schema_version", CCS_IR_SCHEMA_VERSION)),
        )
        expected_digest = str(data.get("digest", "") or "")
        if expected_digest and expected_digest != ir.digest:
            raise ValueError("CCS-IR artifact digest does not match its semantic content")
        return ir

    def to_program(self) -> Program:
        return Program(
            program_id=self.program_id,
            seed=self.seed,
            operations=[node.operation.to_operation() for node in self.nodes],
        )

    def relation(self, relation_id: str) -> RelationIR:
        for relation in self.source_relations:
            if relation.relation_id == relation_id:
                return relation
        for node in self.nodes:
            if node.output_relation.relation_id == relation_id:
                return node.output_relation
        raise KeyError(relation_id)

    def validate(self) -> None:
        if self.schema_version != CCS_IR_SCHEMA_VERSION:
            raise ValueError(f"unsupported CCS relational IR schema: {self.schema_version}")
        known_relations: set[str] = set()
        for relation in self.source_relations:
            _validate_relation(relation)
            if relation.relation_id in known_relations:
                raise ValueError(f"duplicate source relation id: {relation.relation_id}")
            known_relations.add(relation.relation_id)
        if self.primary_relation_id not in known_relations:
            raise ValueError("primary relation is not a source relation")
        node_ids: set[str] = set()
        for expected_index, node in enumerate(self.nodes):
            if node.index != expected_index:
                raise ValueError("CCS-IR nodes must use contiguous indices")
            if node.node_id in node_ids:
                raise ValueError(f"duplicate CCS-IR node id: {node.node_id}")
            unknown_inputs = set(node.input_relations) - known_relations
            if unknown_inputs:
                raise ValueError(f"node {node.node_id} has unknown inputs: {sorted(unknown_inputs)}")
            if node.output_relation.relation_id in known_relations:
                raise ValueError(f"duplicate output relation id: {node.output_relation.relation_id}")
            if node.operation.kind != node.kind:
                raise ValueError(f"node {node.node_id} kind does not match lowering payload")
            _validate_relation(node.output_relation)
            node_ids.add(node.node_id)
            known_relations.add(node.output_relation.relation_id)
        if self.output_relation_id not in known_relations:
            raise ValueError("CCS-IR output relation is unknown")
        obligation_ids: set[str] = set()
        for obligation in self.program_obligations:
            if obligation.obligation_id in obligation_ids:
                raise ValueError(
                    f"duplicate program obligation id: {obligation.obligation_id}"
                )
            unknown_anchors = set(obligation.anchor_node_ids) - node_ids
            if unknown_anchors:
                raise ValueError(
                    f"program obligation {obligation.obligation_id} has unknown anchors: "
                    f"{sorted(unknown_anchors)}"
                )
            unknown_sources = set(obligation.source_relation_ids) - {
                relation.relation_id for relation in self.source_relations
            }
            if unknown_sources:
                raise ValueError(
                    f"program obligation {obligation.obligation_id} has unknown sources: "
                    f"{sorted(unknown_sources)}"
                )
            obligation_ids.add(obligation.obligation_id)


def case_to_ccs_ir(case: Case) -> ContractCarryingRelationalIR:
    if not case.tables:
        raise ValueError("CCS-IR requires at least one source table")
    source_relations = tuple(_source_relation(table, index) for index, table in enumerate(case.tables))
    input_layouts = tuple(
        _input_layout(case, table, relation, index)
        for index, (table, relation) in enumerate(
            zip(case.tables, source_relations, strict=True)
        )
    )
    source_by_table = {
        table.name: relation
        for table, relation in zip(case.tables, source_relations, strict=True)
    }
    table_by_name = {table.name: table for table in case.tables}
    current = source_relations[0]
    state = ProgramState.from_table(case.tables[0])
    nodes: list[RelNodeIR] = []

    for index, operation in enumerate(case.program.operations):
        payload = operation.to_dict()
        kind = op_kind(operation, default="unknown")
        input_relations = _input_relation_ids(current, operation, source_by_table)
        reads = _column_reads(current, operation, source_by_table)
        apply_operation_state(state, operation, tables=table_by_name)
        operation_ir = OperationIR.from_operation(payload)
        node_id = _node_id(index, operation_ir)
        output_columns = _output_columns(
            node_id,
            state,
            current,
            operation,
            reads,
            source_by_table,
        )
        ordering = _output_ordering(kind, operation, current.ordering, output_columns)
        output_relation = RelationIR(
            relation_id=f"rel-{short_canonical_hash({'node_id': node_id, 'schema': [c.to_dict() for c in output_columns]}, 20)}",
            name=f"after:{index}:{kind}",
            columns=output_columns,
            ordering=ordering,
        )
        node = RelNodeIR(
            node_id=node_id,
            index=index,
            kind=kind,
            input_relations=input_relations,
            output_relation=output_relation,
            column_reads=reads,
            scalar_expressions=_scalar_expressions(
                node_id, operation, reads, output_relation
            ),
            aggregates=_aggregate_calls(node_id, operation, reads, output_relation),
            required_capabilities=_required_capabilities(
                kind, operation, reads, output_relation
            ),
            contract=_semantic_contract(kind, operation, output_relation),
            row_effect=_row_effect(kind),
            column_effect=_column_effect(kind),
            order_effect=_order_effect(kind),
            operation=operation_ir,
        )
        nodes.append(node)
        current = output_relation

    required_capabilities = {
        capability for node in nodes for capability in node.required_capabilities
    }
    required_capabilities.add("table:multi" if len(case.tables) > 1 else "table:single")
    required_capabilities.update(
        f"type:{column.logical_type}"
        for relation in source_relations
        for column in relation.columns
        if column.logical_type and column.logical_type != "unknown"
    )
    if any(
        column.nullable
        for relation in source_relations
        for column in relation.columns
    ):
        required_capabilities.add("nulls")
    required = tuple(sorted(required_capabilities))
    program_obligations = infer_program_obligations(
        case,
        node_ids=tuple(node.node_id for node in nodes),
        source_relation_ids=tuple(
            relation.relation_id for relation in source_relations
        ),
    )
    testing_objectives = tuple(
        sorted(
            {
                obligation.obligation_id
                for node in nodes
                for obligation in node.contract.test_obligations
            }
            | {
                obligation.obligation_id
                for obligation in program_obligations
            }
        )
    )
    return ContractCarryingRelationalIR(
        program_id=case.program.program_id,
        seed=case.program.seed,
        primary_relation_id=source_relations[0].relation_id,
        source_relations=source_relations,
        input_layouts=input_layouts,
        nodes=tuple(nodes),
        program_obligations=program_obligations,
        output_relation_id=current.relation_id,
        required_capabilities=required,
        testing_objectives=testing_objectives,
    )


def _column_from_dict(data: Mapping[str, Any]) -> ColumnIR:
    return ColumnIR(
        column_id=str(data.get("column_id", "")),
        name=str(data.get("name", "")),
        logical_type=str(data.get("logical_type", "unknown")),
        nullable=bool(data.get("nullable", False)),
        lineage=tuple(str(item) for item in data.get("lineage", ())),
    )


def _column_ref_from_dict(data: Mapping[str, Any]) -> ColumnRefIR:
    return ColumnRefIR(
        relation_id=str(data.get("relation_id", "")),
        column_id=str(data.get("column_id", "")),
        name=str(data.get("name", "")),
        logical_type=str(data.get("logical_type", "unknown")),
        nullable=bool(data.get("nullable", False)),
    )


def _ordering_from_dict(data: Mapping[str, Any]) -> OrderingIR:
    return OrderingIR(
        mode=str(data.get("mode", "none")),
        observed=bool(data.get("observed", False)),
        keys=tuple(
            OrderKeyIR(
                column_id=str(item.get("column_id", "")),
                column_name=str(item.get("column_name", "")),
                ascending=bool(item.get("ascending", True)),
                nulls=str(item.get("nulls", "last")),
            )
            for item in data.get("keys", ())
            if isinstance(item, Mapping)
        ),
    )


def _relation_from_dict(data: Mapping[str, Any]) -> RelationIR:
    ordering = data.get("ordering", {})
    return RelationIR(
        relation_id=str(data.get("relation_id", "")),
        name=str(data.get("name", "")),
        columns=tuple(
            _column_from_dict(item)
            for item in data.get("columns", ())
            if isinstance(item, Mapping)
        ),
        ordering=_ordering_from_dict(ordering if isinstance(ordering, Mapping) else {}),
    )


def _input_layout_from_dict(data: Mapping[str, Any]) -> InputLayoutIR:
    attributes = data.get("attributes", {})
    return InputLayoutIR(
        relation_id=str(data.get("relation_id", "")),
        table_name=str(data.get("table_name", "")),
        representation=str(data.get("representation", "logical_rows")),
        row_count=int(data.get("row_count", 0)),
        chunk_count=(
            None if data.get("chunk_count") is None else int(data.get("chunk_count", 0))
        ),
        dictionary_columns=tuple(
            str(item) for item in data.get("dictionary_columns", ())
        ),
        attributes=CanonicalPayload.build(
            attributes if isinstance(attributes, Mapping) else {}
        ),
    )


def _node_from_dict(data: Mapping[str, Any]) -> RelNodeIR:
    output_relation = data.get("output_relation", {})
    operation = data.get("operation", {})
    contract = data.get("contract", {})
    return RelNodeIR(
        node_id=str(data.get("node_id", "")),
        index=int(data.get("index", 0)),
        kind=str(data.get("kind", "unknown")),
        input_relations=tuple(str(item) for item in data.get("input_relations", ())),
        output_relation=_relation_from_dict(
            output_relation if isinstance(output_relation, Mapping) else {}
        ),
        column_reads=tuple(
            _column_ref_from_dict(item)
            for item in data.get("column_reads", ())
            if isinstance(item, Mapping)
        ),
        scalar_expressions=tuple(
            _scalar_expression_from_dict(item)
            for item in data.get("scalar_expressions", ())
            if isinstance(item, Mapping)
        ),
        aggregates=tuple(
            _aggregate_from_dict(item)
            for item in data.get("aggregates", ())
            if isinstance(item, Mapping)
        ),
        required_capabilities=tuple(
            str(item) for item in data.get("required_capabilities", ())
        ),
        contract=_contract_from_dict(contract if isinstance(contract, Mapping) else {}),
        row_effect=str(data.get("row_effect", "preserving")),
        column_effect=str(data.get("column_effect", "preserve")),
        order_effect=str(data.get("order_effect", "preserve")),
        operation=OperationIR(
            kind=str(operation.get("kind", data.get("kind", "unknown"))),
            arguments=CanonicalPayload.build(
                operation.get("arguments", {})
                if isinstance(operation, Mapping)
                and isinstance(operation.get("arguments", {}), Mapping)
                else {}
            ),
        ),
    )


def _scalar_expression_from_dict(data: Mapping[str, Any]) -> ScalarExpressionIR:
    attributes = data.get("attributes", {})
    return ScalarExpressionIR(
        expression_id=str(data.get("expression_id", "")),
        kind=str(data.get("kind", "unknown")),
        inputs=tuple(
            _column_ref_from_dict(item)
            for item in data.get("inputs", ())
            if isinstance(item, Mapping)
        ),
        result_type=str(data.get("result_type", "unknown")),
        nullable=bool(data.get("nullable", False)),
        attributes=CanonicalPayload.build(
            attributes if isinstance(attributes, Mapping) else {}
        ),
    )


def _aggregate_from_dict(data: Mapping[str, Any]) -> AggregateCallIR:
    input_column = data.get("input_column")
    return AggregateCallIR(
        aggregate_id=str(data.get("aggregate_id", "")),
        function=str(data.get("function", "unknown")),
        input_column=(
            _column_ref_from_dict(input_column)
            if isinstance(input_column, Mapping)
            else None
        ),
        output_column_id=str(data.get("output_column_id", "")),
        output_name=str(data.get("output_name", "")),
        result_type=str(data.get("result_type", "unknown")),
        nullable=bool(data.get("nullable", False)),
    )


def _contract_from_dict(data: Mapping[str, Any]) -> SemanticContractIR:
    return SemanticContractIR(
        equivalence=str(data.get("equivalence", "bag")),
        definedness=str(data.get("definedness", "defined")),
        order_observable=bool(data.get("order_observable", False)),
        duplicate_sensitive=bool(data.get("duplicate_sensitive", True)),
        null_sensitive=bool(data.get("null_sensitive", False)),
        numeric_policy=str(data.get("numeric_policy", "exact_lossless")),
        error_policy=str(data.get("error_policy", "typed_error_category")),
        preconditions=tuple(str(item) for item in data.get("preconditions", ())),
        postconditions=tuple(str(item) for item in data.get("postconditions", ())),
        test_obligations=tuple(
            TestObligationIR(
                obligation_id=str(item.get("obligation_id", "")),
                kind=str(item.get("kind", "fault_model")),
                transformation=str(item.get("transformation", "identity")),
                expected_relation=str(item.get("expected_relation", "")),
                preconditions=tuple(
                    str(value) for value in item.get("preconditions", ())
                ),
                oracle=str(item.get("oracle", "differential")),
                target_fault_models=tuple(
                    str(value) for value in item.get("target_fault_models", ())
                ),
                estimated_cost=float(item.get("estimated_cost", 1.0)),
            )
            for item in data.get("test_obligations", ())
            if isinstance(item, Mapping)
        ),
    )


def _canonical_operation_arguments(kind: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(arguments))
    if kind != "sort":
        return payload
    sort_operation = {"op": kind, **payload}
    payload.pop("columns", None)
    payload.pop("ascending", None)
    payload.pop("keys", None)
    payload["keys"] = [key.to_dict() for key in normalize_sort_keys(sort_operation)]
    return payload


def _source_relation(table: TableData, index: int) -> RelationIR:
    relation_id = f"rel-source-{short_canonical_hash({'index': index, 'table': table.name}, 16)}"
    columns = tuple(
        ColumnIR(
            column_id=f"col-{short_canonical_hash({'relation': relation_id, 'index': column_index, 'name': column.name}, 20)}",
            name=column.name,
            logical_type=column.type,
            nullable=bool(
                column.nullable
                or any(row.get(column.name) is None for row in table.rows if column.name in row)
            ),
        )
        for column_index, column in enumerate(table.columns)
    )
    return RelationIR(
        relation_id=relation_id,
        name=table.name,
        columns=columns,
        ordering=OrderingIR(mode="input", observed=False),
    )


def _input_layout(
    case: Case,
    table: TableData,
    relation: RelationIR,
    index: int,
) -> InputLayoutIR:
    layout_map = case.metadata.get("input_layouts", {}) if isinstance(case.metadata, dict) else {}
    raw_layout: Mapping[str, Any] = {}
    if isinstance(layout_map, Mapping):
        candidate = layout_map.get(table.name, layout_map.get(str(index), {}))
        if isinstance(candidate, Mapping):
            raw_layout = candidate
    representation = str(raw_layout.get("representation", "logical_rows") or "logical_rows")
    chunk_count_value = raw_layout.get("chunk_count")
    chunk_count = None if chunk_count_value is None else max(0, int(chunk_count_value))
    dictionary_columns = tuple(
        str(name)
        for name in raw_layout.get("dictionary_columns", ())
        if str(name)
    )
    known_fields = {"representation", "chunk_count", "dictionary_columns"}
    attributes = {
        str(key): value for key, value in raw_layout.items() if str(key) not in known_fields
    }
    return InputLayoutIR(
        relation_id=relation.relation_id,
        table_name=table.name,
        representation=representation,
        row_count=len(table.rows),
        chunk_count=chunk_count,
        dictionary_columns=dictionary_columns,
        attributes=CanonicalPayload.build(attributes),
    )


def _input_relation_ids(
    current: RelationIR,
    operation: Mapping[str, Any],
    source_by_table: Mapping[str, RelationIR],
) -> tuple[str, ...]:
    relation_ids = [current.relation_id]
    table_name = op_table(operation)
    if table_name and table_name in source_by_table:
        relation_ids.append(source_by_table[table_name].relation_id)
    return tuple(unique_preserve_order(relation_ids))


def _column_reads(
    current: RelationIR,
    operation: Mapping[str, Any],
    source_by_table: Mapping[str, RelationIR],
) -> tuple[ColumnRefIR, ...]:
    kind = op_kind(operation)
    current_names: list[str] = []
    right_names: list[str] = []
    if kind == "filter":
        current_names = [op_column(operation)]
    elif kind == "tuple_absence_filter":
        current_names = op_columns(operation)
        right_names = op_right_columns(operation)
    elif kind in {"drop_nulls", "select", "distinct"}:
        current_names = op_columns(operation)
    elif kind == "fill_null":
        current_names = [op_column(operation)]
    elif kind == "coalesce":
        current_names = coalesce_sources(operation)
    elif kind == "case_when":
        current_names = [condition_column(operation)]
    elif kind == "mutate":
        current_names = [expr_source(operation), expr_other(operation), expr_numerator(operation)]
    elif kind == "row_number_filter":
        current_names = [
            *op_partition_columns(operation),
            *(key.column for key in normalized_order_by_keys(operation)),
        ]
    elif kind == "running_sum":
        current_names = [
            str(operation.get("source", "")),
            *op_partition_columns(operation),
            *(key.column for key in normalized_order_by_keys(operation)),
        ]
    elif kind == "sortedness_check":
        current_names = [op_column(operation)]
    elif kind in {"join", "semi_join", "anti_join"}:
        current_names = join_left_keys(operation)
        right_names = join_right_keys(operation)
    elif kind == "union_all":
        current_names = [column.name for column in current.columns]
        right_names = list(current_names)
    elif kind == "groupby":
        current_names = [
            *groupby_keys(operation),
            *(aggregate_column(aggregate) for aggregate in aggregate_specs(operation)),
        ]
    elif kind == "aggregate":
        current_names = [aggregate_column(aggregate) for aggregate in aggregate_specs(operation)]
    elif kind == "sort":
        current_names = [key.column for key in normalize_sort_keys(operation)]
    elif kind.endswith("_probe"):
        current_names = _generic_probe_columns(operation)

    refs = _refs_for_names(current, current_names)
    table_name = op_table(operation)
    right = source_by_table.get(table_name)
    if right is not None:
        refs.extend(_refs_for_names(right, right_names))
    return tuple(_unique_refs(refs))


def _output_columns(
    node_id: str,
    state: ProgramState,
    current: RelationIR,
    operation: Mapping[str, Any],
    reads: tuple[ColumnRefIR, ...],
    source_by_table: Mapping[str, RelationIR],
) -> tuple[ColumnIR, ...]:
    kind = op_kind(operation)
    right = source_by_table.get(op_table(operation))
    columns: list[ColumnIR] = []
    for position, name in enumerate(state.columns):
        lineage = _output_lineage(name, kind, operation, current, right, reads)
        logical_type = str(state.column_types.get(name, "unknown") or "unknown")
        nullable = name in state.nullable_columns
        column_id = f"col-{short_canonical_hash({'node': node_id, 'position': position, 'name': name, 'type': logical_type, 'nullable': nullable, 'lineage': lineage}, 20)}"
        columns.append(
            ColumnIR(
                column_id=column_id,
                name=name,
                logical_type=logical_type,
                nullable=nullable,
                lineage=lineage,
            )
        )
    return tuple(columns)


def _output_lineage(
    name: str,
    kind: str,
    operation: Mapping[str, Any],
    current: RelationIR,
    right: RelationIR | None,
    reads: tuple[ColumnRefIR, ...],
) -> tuple[str, ...]:
    current_column = current.column(name)
    if kind in {"groupby", "aggregate"}:
        aggregate = next(
            (item for item in aggregate_specs(operation) if aggregate_alias(item) == name),
            None,
        )
        if aggregate is not None:
            return _lineage_ids(reads, [aggregate_column(aggregate)])
    if kind == "mutate" and name == op_column(operation):
        return _lineage_ids(reads, [expr_source(operation), expr_other(operation), expr_numerator(operation)])
    if kind == "coalesce" and name == op_output_alias(operation):
        return _lineage_ids(reads, coalesce_sources(operation))
    if kind == "case_when" and name == op_output_alias(operation):
        return _lineage_ids(reads, [condition_column(operation)])
    if kind == "running_sum" and name == op_column(operation):
        return _lineage_ids(reads, [str(operation.get("source", ""))])
    if kind == "fill_null" and name == op_column(operation) and current_column is not None:
        return (current_column.column_id,)
    if current_column is not None:
        return (current_column.column_id,)
    right_column = right.column(name) if right is not None else None
    if right_column is not None:
        return (right_column.column_id,)
    return tuple(ref.column_id for ref in reads)


def _scalar_expressions(
    node_id: str,
    operation: Mapping[str, Any],
    reads: tuple[ColumnRefIR, ...],
    output: RelationIR,
) -> tuple[ScalarExpressionIR, ...]:
    kind = op_kind(operation)
    specs: list[tuple[str, Mapping[str, Any], str, bool]] = []
    if kind == "mutate":
        out = output.column(op_column(operation))
        if out is not None:
            specs.append((expr_kind(operation, default="unknown"), expr_payload(operation), out.logical_type, out.nullable))
    if kind in {"filter", "case_when"}:
        condition = operation.get("condition", operation)
        payload = condition if isinstance(condition, Mapping) else {}
        specs.append((f"predicate:{condition_cmp(operation, default='unknown')}", payload, "bool", True))
    return tuple(
        ScalarExpressionIR(
            expression_id=f"expr-{short_canonical_hash({'node': node_id, 'index': index, 'kind': expr_kind_value, 'payload': dict(payload)}, 20)}",
            kind=expr_kind_value,
            inputs=reads,
            result_type=result_type,
            nullable=nullable,
            attributes=CanonicalPayload.build(payload),
        )
        for index, (expr_kind_value, payload, result_type, nullable) in enumerate(specs)
    )


def _aggregate_calls(
    node_id: str,
    operation: Mapping[str, Any],
    reads: tuple[ColumnRefIR, ...],
    output: RelationIR,
) -> tuple[AggregateCallIR, ...]:
    calls: list[AggregateCallIR] = []
    for index, aggregate in enumerate(aggregate_specs(operation)):
        alias = aggregate_alias(aggregate)
        output_column = output.column(alias)
        if output_column is None:
            continue
        input_name = aggregate_column(aggregate)
        input_ref = next((ref for ref in reads if ref.name == input_name), None)
        function = aggregate_func(aggregate, default="unknown")
        calls.append(
            AggregateCallIR(
                aggregate_id=f"agg-{short_canonical_hash({'node': node_id, 'index': index, 'function': function, 'input': input_name, 'output': alias}, 20)}",
                function=function,
                input_column=input_ref,
                output_column_id=output_column.column_id,
                output_name=alias,
                result_type=output_column.logical_type,
                nullable=output_column.nullable,
            )
        )
    return tuple(calls)


def _required_capabilities(
    kind: str,
    operation: Mapping[str, Any],
    reads: tuple[ColumnRefIR, ...],
    output: RelationIR,
) -> tuple[str, ...]:
    capabilities = {f"op:{kind}"}
    if kind == "mutate":
        capabilities.add(f"expr:{expr_kind(operation, default='unknown')}")
    capabilities.update(
        f"agg:{aggregate_func(aggregate)}"
        for aggregate in aggregate_specs(operation)
        if aggregate_func(aggregate)
    )
    capabilities.update(
        f"type:{column.logical_type}"
        for column in [*reads, *output.columns]
        if column.logical_type and column.logical_type != "unknown"
    )
    if any(column.nullable for column in [*reads, *output.columns]):
        capabilities.add("nulls")
    return tuple(sorted(capabilities))


def _semantic_contract(
    kind: str,
    operation: Mapping[str, Any],
    output: RelationIR,
) -> SemanticContractIR:
    order_observable = output.ordering.observed or kind in {
        "sort",
        "limit",
        "offset",
        "running_sum",
        "row_number_filter",
        "sortedness_check",
    }
    if kind.endswith("_probe") or kind == "sortedness_check":
        equivalence = "scalar_exact"
    elif kind == "distinct":
        equivalence = "set"
    elif order_observable:
        equivalence = "ordered"
    else:
        equivalence = "bag"
    preconditions: list[str] = ["input_schema_valid"]
    if kind in {"join", "semi_join", "anti_join", "tuple_absence_filter", "union_all"}:
        preconditions.append("secondary_relation_available")
    if kind in {"sort", "running_sum", "row_number_filter"}:
        preconditions.append("order_keys_resolvable")
    postconditions = [f"row_effect:{_row_effect(kind)}", f"column_effect:{_column_effect(kind)}"]
    if output.ordering.mode != "none":
        postconditions.append(f"ordering:{output.ordering.mode}")
    obligations = _test_obligations(
        metamorphic_relations=_metamorphic_relations(kind, operation),
        fault_models=_fault_models(kind, operation),
        preconditions=tuple(preconditions),
    )
    return SemanticContractIR(
        equivalence=equivalence,
        definedness=(
            "backend_extension"
            if kind.endswith("_probe")
            else "unspecified"
            if kind == "unknown"
            else "defined"
        ),
        order_observable=order_observable,
        duplicate_sensitive=kind != "distinct",
        null_sensitive=any(column.nullable for column in output.columns),
        numeric_policy="exact_lossless",
        error_policy="typed_error_category",
        preconditions=tuple(preconditions),
        postconditions=tuple(postconditions),
        test_obligations=obligations,
    )


def _test_obligations(
    *,
    metamorphic_relations: Sequence[str],
    fault_models: Sequence[str],
    preconditions: tuple[str, ...],
) -> tuple[TestObligationIR, ...]:
    metamorphic = (
        TestObligationIR(
            obligation_id=relation,
            kind="metamorphic_relation",
            transformation=relation,
            expected_relation="contract_equivalent",
            preconditions=preconditions,
            oracle="metamorphic",
            target_fault_models=tuple(fault_models),
            estimated_cost=_metamorphic_obligation_cost(relation),
        )
        for relation in metamorphic_relations
    )
    faults = (
        TestObligationIR(
            obligation_id=model,
            kind="fault_model",
            transformation="identity",
            expected_relation="potential_semantic_divergence",
            preconditions=preconditions,
            oracle="differential_or_witness",
            target_fault_models=(model,),
            estimated_cost=0.0,
        )
        for model in fault_models
    )
    return (*metamorphic, *faults)


def _metamorphic_obligation_cost(relation: str) -> float:
    if relation in {
        "row_permutation",
        "join_unmatched_dimension_injection",
        "join_inner_left_equivalence",
    }:
        return 1.5
    if relation.endswith("input_materialization"):
        return 1.25
    return 1.0


def _metamorphic_relations(
    kind: str,
    operation: Mapping[str, Any],
) -> tuple[str, ...]:
    relations = {
        "filter": ("filter_idempotence", "filter_idempotence_immediate"),
        "drop_nulls": ("drop_nulls_idempotence",),
        "fill_null": ("fill_null_idempotence",),
        "coalesce": ("coalesce_idempotence",),
        "case_when": ("case_when_idempotence",),
        "distinct": ("distinct_idempotence",),
        "sort": ("sort_idempotence",),
        "limit": ("limit_idempotence",),
        "select": ("select_idempotence",),
    }
    declared = list(relations.get(kind, ()))
    if kind == "groupby" and len(groupby_keys(operation)) >= 2:
        declared.append("groupby_key_permutation")
    if kind == "mutate":
        string_relations = {
            "string_lower": "string_lower_idempotence",
            "string_upper": "string_upper_idempotence",
            "string_strip": "string_strip_idempotence",
            "string_null_if_empty": "string_null_if_empty_idempotence",
            "string_replace": "string_replace_idempotence",
            "string_slice": "string_slice_prefix_idempotence",
            "string_split_part": "string_split_part_idempotence",
        }
        relation = string_relations.get(expr_kind(operation, default=""))
        if relation:
            declared.append(relation)
    return tuple(declared)


def _fault_models(kind: str, operation: Mapping[str, Any]) -> tuple[str, ...]:
    models: set[str] = {f"operation:{kind}"}
    if kind in {"filter", "drop_nulls", "fill_null", "coalesce", "case_when"}:
        models.add("null_semantics")
    if kind in {"join", "semi_join", "anti_join", "tuple_absence_filter"}:
        models.update({"join_cardinality", "join_null_semantics"})
    if kind in {"groupby", "aggregate", "running_sum"}:
        models.update({"aggregation_null_semantics", "numeric_precision"})
    if kind in {"sort", "limit", "offset", "running_sum", "row_number_filter"}:
        models.update({"ordering_stability", "null_placement"})
    if kind == "mutate":
        expression_kind = expr_kind(operation)
        if expression_kind == "cast":
            models.update({"cast_boundary", "dtype_lowering"})
        if expression_kind in {"arith_const", "reverse_division_columns"}:
            models.update({"numeric_precision", "signed_zero"})
        if expression_kind.startswith("string_"):
            models.add("string_semantics")
    return tuple(sorted(models))


def _output_ordering(
    kind: str,
    operation: Mapping[str, Any],
    previous: OrderingIR,
    output_columns: Sequence[ColumnIR],
) -> OrderingIR:
    output_by_name = {column.name: column for column in output_columns}
    if kind == "sort":
        return OrderingIR(
            mode="explicit",
            observed=True,
            keys=_order_keys(normalize_sort_keys(operation), output_by_name),
        )
    if kind in {"running_sum", "row_number_filter"}:
        return OrderingIR(
            mode="explicit",
            observed=True,
            keys=_order_keys(normalized_order_by_keys(operation), output_by_name),
        )
    if kind in {"limit", "offset", "sortedness_check"}:
        return OrderingIR(mode=previous.mode, observed=True, keys=_remap_order_keys(previous.keys, output_by_name))
    if kind in {
        "filter",
        "tuple_absence_filter",
        "drop_nulls",
        "fill_null",
        "coalesce",
        "case_when",
        "mutate",
        "select",
        "semi_join",
        "anti_join",
    }:
        return OrderingIR(
            mode=previous.mode,
            observed=previous.observed,
            keys=_remap_order_keys(previous.keys, output_by_name),
        )
    return OrderingIR()


def _order_keys(
    keys: Sequence[SortKey],
    output_by_name: Mapping[str, ColumnIR],
) -> tuple[OrderKeyIR, ...]:
    return tuple(
        OrderKeyIR(
            column_id=output_by_name[key.column].column_id,
            column_name=key.column,
            ascending=key.ascending,
            nulls=key.nulls,
        )
        for key in keys
        if key.column in output_by_name
    )


def _remap_order_keys(
    keys: Sequence[OrderKeyIR],
    output_by_name: Mapping[str, ColumnIR],
) -> tuple[OrderKeyIR, ...]:
    return tuple(
        OrderKeyIR(
            column_id=output_by_name[key.column_name].column_id,
            column_name=key.column_name,
            ascending=key.ascending,
            nulls=key.nulls,
        )
        for key in keys
        if key.column_name in output_by_name
    )


def _node_id(index: int, operation: OperationIR) -> str:
    return (
        f"node-{index:03d}-{operation.kind}-"
        f"{short_canonical_hash({'index': index, 'operation': operation.to_dict()}, 12)}"
    )


def _refs_for_names(relation: RelationIR, names: Sequence[str]) -> list[ColumnRefIR]:
    refs: list[ColumnRefIR] = []
    for name in names:
        column = relation.column(str(name or ""))
        if column is not None:
            refs.append(ColumnRefIR.from_column(relation.relation_id, column))
    return refs


def _unique_refs(refs: Sequence[ColumnRefIR]) -> list[ColumnRefIR]:
    out: list[ColumnRefIR] = []
    seen: set[tuple[str, str]] = set()
    for ref in refs:
        key = (ref.relation_id, ref.column_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _lineage_ids(reads: Sequence[ColumnRefIR], names: Sequence[str]) -> tuple[str, ...]:
    wanted = {str(name) for name in names if str(name)}
    return tuple(ref.column_id for ref in reads if ref.name in wanted)


def _generic_probe_columns(operation: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for key in ("column", "source"):
        value = operation.get(key)
        if isinstance(value, str) and value:
            names.append(value)
    for key in ("columns", "partition_by"):
        value = operation.get(key)
        if isinstance(value, (list, tuple)):
            names.extend(str(item) for item in value if str(item))
    return unique_preserve_order(names)


def _row_effect(kind: str) -> str:
    if kind in {"filter", "drop_nulls", "semi_join", "anti_join", "tuple_absence_filter", "limit", "offset", "row_number_filter"}:
        return "reducing"
    if kind in {"join", "union_all"}:
        return "expanding"
    if kind in {"groupby", "aggregate", "distinct"}:
        return "collapsing"
    if kind.endswith("_probe") or kind == "sortedness_check":
        return "scalarizing"
    return "preserving"


def _column_effect(kind: str) -> str:
    if kind in {"mutate", "coalesce", "case_when", "running_sum", "join"}:
        return "upsert"
    if kind in {"select", "distinct", "groupby", "aggregate"}:
        return "project"
    if kind.endswith("_probe") or kind == "sortedness_check":
        return "replace"
    return "preserve"


def _order_effect(kind: str) -> str:
    if kind in {"sort", "running_sum", "row_number_filter"}:
        return "define_and_observe"
    if kind in {"limit", "offset", "sortedness_check"}:
        return "observe"
    if kind in {"join", "union_all", "distinct", "groupby", "aggregate"} or kind.endswith("_probe"):
        return "discard"
    return "preserve"


def _validate_relation(relation: RelationIR) -> None:
    column_ids = [column.column_id for column in relation.columns]
    column_names = [column.name for column in relation.columns]
    if len(column_ids) != len(set(column_ids)):
        raise ValueError(f"relation {relation.relation_id} contains duplicate column ids")
    if len(column_names) != len(set(column_names)):
        raise ValueError(f"relation {relation.relation_id} contains duplicate column names")
    known_ids = set(column_ids)
    unknown_order_ids = {key.column_id for key in relation.ordering.keys} - known_ids
    if unknown_order_ids:
        raise ValueError(
            f"relation {relation.relation_id} ordering references unknown columns: {sorted(unknown_order_ids)}"
        )
