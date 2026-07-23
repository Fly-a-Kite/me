from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

from datadiff.canonicalization import short_canonical_hash
from datadiff.ccs_ir import ContractCarryingRelationalIR, TestObligationIR
from datadiff.dsl import Case
from datadiff.metamorphic import (
    MetamorphicVariant,
    build_metamorphic_variants_for_relations,
    metamorphic_relation_builder_registry,
)
from datadiff.obligation_priority import (
    COMPLETE_BUILDER_ORDER,
    ObligationPriorityInput,
    ObligationPriorityPlan,
    build_obligation_priority_plan,
)
from datadiff.program_obligations import ProgramObligationIR


TEST_OBLIGATION_REGISTRY_SCHEMA_VERSION = "ccs-test-obligation-registry-v3"
ObligationIR = TestObligationIR | ProgramObligationIR


@dataclass(frozen=True, slots=True)
class ObligationDeclaration:
    obligation: ObligationIR
    declaration_scopes: tuple[str, ...]
    anchor_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExecutableObligation:
    obligation: ObligationIR
    declaration_scopes: tuple[str, ...]
    anchor_node_ids: tuple[str, ...]
    registered: bool
    applicable: bool
    variant_count: int
    priority: dict[str, Any]
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.obligation.to_dict(),
            "declaration_scopes": list(self.declaration_scopes),
            "anchor_node_ids": list(self.anchor_node_ids),
            "registered": self.registered,
            "applicable": self.applicable,
            "variant_count": self.variant_count,
            "priority": dict(self.priority),
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True, slots=True)
class ObligationSelection:
    variants: tuple[MetamorphicVariant, ...]
    obligations: tuple[ExecutableObligation, ...]
    fault_models: tuple[str, ...]
    registry_digest: str
    priority_plan: ObligationPriorityPlan

    def to_dict(self) -> dict[str, Any]:
        applicable = [item for item in self.obligations if item.applicable]
        node_declarations = [
            item
            for item in self.obligations
            if "node_contract" in item.declaration_scopes
        ]
        program_declarations = [
            item
            for item in self.obligations
            if "program" in item.declaration_scopes
        ]
        return {
            "schema_version": TEST_OBLIGATION_REGISTRY_SCHEMA_VERSION,
            "mode": "ccs_guided",
            "registry_digest": self.registry_digest,
            "priority": self.priority_plan.to_dict(),
            "declared_obligation_count": len(self.obligations),
            "node_contract_obligation_count": len(node_declarations),
            "program_obligation_count": len(program_declarations),
            "applicable_obligation_count": len(applicable),
            "constructed_variant_count": len(self.variants),
            "estimated_execution_cost": sum(
                item.obligation.estimated_cost for item in applicable
            ),
            "fault_models": list(self.fault_models),
            "obligations": [item.to_dict() for item in self.obligations],
            "variant_names": [variant.name for variant in self.variants],
            "relations": list(dict.fromkeys(variant.relation for variant in self.variants)),
        }


@lru_cache(maxsize=1)
def executable_obligation_registry() -> dict[str, Any]:
    builders = metamorphic_relation_builder_registry()
    payload = {
        "schema_version": TEST_OBLIGATION_REGISTRY_SCHEMA_VERSION,
        "relation_order": list(builders),
        "relations": {
            relation: {
                "transformation": relation,
                "builders": [builder.__name__ for builder in relation_builders],
                "oracle": "metamorphic",
            }
            for relation, relation_builders in sorted(builders.items())
        },
        "fault_model_executor": "base_differential_or_witness",
    }
    return {
        **payload,
        "digest": f"obligation-registry-{short_canonical_hash(payload, 64)}",
    }


def select_executable_test_obligations(
    case: Case,
    ir: ContractCarryingRelationalIR,
    *,
    relation_order: Iterable[str] = (),
    priority_mode: str = COMPLETE_BUILDER_ORDER,
) -> ObligationSelection:
    registry = executable_obligation_registry()
    registered_relations = set(registry["relations"])
    declarations = _metamorphic_declarations(ir)
    priority_plan = build_obligation_priority_plan(
        [
            _priority_input(declaration, ir=ir, case=case)
            for declaration in declarations.values()
        ],
        mode=priority_mode,
        complete_builder_order=registry["relation_order"],
        semantic_digest=ir.semantic_digest,
        preferred_order=relation_order,
    )
    ordered_relations = list(priority_plan.relation_order)
    priority_by_relation = {
        score.obligation_id: score.to_dict()
        for score in priority_plan.scores
    }
    eligible_relations = [
        relation
        for relation in ordered_relations
        if relation in registered_relations
        and _preconditions_satisfied(
            case, declarations[relation].obligation.preconditions
        )
    ]
    variants = build_metamorphic_variants_for_relations(case, eligible_relations)
    variants.sort(key=_variant_order_key(ordered_relations))
    variant_counts: dict[str, int] = {}
    for variant in variants:
        variant_counts[variant.relation] = variant_counts.get(variant.relation, 0) + 1

    obligations = tuple(
        _executable_obligation(
            case,
            relation,
            declarations[relation],
            registered=relation in registered_relations,
            variant_count=variant_counts.get(relation, 0),
            priority=priority_by_relation[relation],
        )
        for relation in ordered_relations
    )
    fault_models = tuple(
        dict.fromkeys(
            [
                obligation.obligation_id
                for node in ir.nodes
                for obligation in node.contract.test_obligations
                if obligation.kind == "fault_model"
            ]
            + [
                fault_model
                for obligation in ir.program_obligations
                for fault_model in obligation.target_fault_models
            ]
        )
    )
    return ObligationSelection(
        variants=tuple(variants),
        obligations=obligations,
        fault_models=fault_models,
        registry_digest=str(registry["digest"]),
        priority_plan=priority_plan,
    )


def _metamorphic_declarations(
    ir: ContractCarryingRelationalIR,
) -> dict[str, ObligationDeclaration]:
    declarations: dict[str, ObligationDeclaration] = {}

    def add(
        obligation: ObligationIR,
        *,
        scope: str,
        anchor_node_ids: tuple[str, ...],
    ) -> None:
        obligation_id = obligation.obligation_id
        existing = declarations.get(obligation_id)
        if existing is None:
            declarations[obligation_id] = ObligationDeclaration(
                obligation=obligation,
                declaration_scopes=(scope,),
                anchor_node_ids=anchor_node_ids,
            )
            return
        declarations[obligation_id] = ObligationDeclaration(
            obligation=existing.obligation,
            declaration_scopes=tuple(
                dict.fromkeys((*existing.declaration_scopes, scope))
            ),
            anchor_node_ids=tuple(
                dict.fromkeys((*existing.anchor_node_ids, *anchor_node_ids))
            ),
        )

    for node in ir.nodes:
        for obligation in node.contract.test_obligations:
            if obligation.kind != "metamorphic_relation":
                continue
            add(
                obligation,
                scope="node_contract",
                anchor_node_ids=(node.node_id,),
            )
    for obligation in ir.program_obligations:
        add(
            obligation,
            scope="program",
            anchor_node_ids=obligation.anchor_node_ids,
        )
    return declarations


def _priority_input(
    declaration: ObligationDeclaration,
    *,
    ir: ContractCarryingRelationalIR,
    case: Case,
) -> ObligationPriorityInput:
    obligation = declaration.obligation
    program_scope = (
        (obligation.scope,)
        if isinstance(obligation, ProgramObligationIR)
        else ()
    )
    source_relation_count = (
        len(obligation.source_relation_ids)
        if isinstance(obligation, ProgramObligationIR)
        else 0
    )
    interaction_features, plan_transition_features = _priority_v2_features(
        declaration,
        ir=ir,
        case=case,
    )
    return ObligationPriorityInput(
        obligation_id=obligation.obligation_id,
        target_fault_models=obligation.target_fault_models,
        estimated_cost=obligation.estimated_cost,
        scopes=tuple(
            dict.fromkeys((*declaration.declaration_scopes, *program_scope))
        ),
        anchor_node_count=len(declaration.anchor_node_ids),
        source_relation_count=source_relation_count,
        interaction_features=interaction_features,
        plan_transition_features=plan_transition_features,
    )


def _priority_v2_features(
    declaration: ObligationDeclaration,
    *,
    ir: ContractCarryingRelationalIR,
    case: Case,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    anchor_ids = set(declaration.anchor_node_ids)
    selected_nodes = [
        node for node in ir.nodes
        if not anchor_ids or node.node_id in anchor_ids
    ]
    selected_kinds = {node.kind for node in selected_nodes}
    all_kinds = [node.kind for node in ir.nodes]
    interaction: set[str] = set()
    transitions: set[str] = set()
    nullable = any(
        column.nullable
        for node in selected_nodes
        for column in (*node.column_reads, *node.output_relation.columns)
    )
    has_cast = any(
        expression.kind == "cast"
        for node in selected_nodes
        for expression in node.scalar_expressions
    )
    numeric = any(
        column.logical_type in {"int", "float"}
        for node in selected_nodes
        for column in (*node.column_reads, *node.output_relation.columns)
    )
    program_kinds = set(all_kinds)
    relevant_kinds = selected_kinds if anchor_ids else program_kinds
    if relevant_kinds.intersection({"sort", "limit", "offset", "running_sum", "row_number_filter"}) \
        and program_kinds.intersection({"limit", "offset"}):
        interaction.add("order_limit_offset")
    if nullable and relevant_kinds.intersection({"groupby", "aggregate"}) \
        and program_kinds.intersection({"sort", "limit"}):
        interaction.add("null_aggregation_order")
    if program_kinds.intersection({"join", "semi_join", "anti_join"}) \
        and "filter" in program_kinds \
        and program_kinds.intersection({"limit", "offset"}):
        interaction.add("join_predicate_limit")
    if has_cast and program_kinds.intersection({"semi_join", "anti_join", "join"}) \
        and program_kinds.intersection({"running_sum", "row_number_filter"}):
        interaction.add("cast_membership_window")
    if program_kinds.intersection({"union_all", "distinct"}) \
        and program_kinds.intersection({"running_sum", "row_number_filter"}):
        interaction.add("set_bag_window")
    if nullable and program_kinds.intersection({"semi_join", "anti_join", "join"}):
        interaction.add("nullable_membership")
    if numeric and program_kinds.intersection({"running_sum", "row_number_filter"}):
        interaction.add("numeric_window")
    if any(
        layout.representation != "logical_rows"
        or (layout.chunk_count is not None and layout.chunk_count > 1)
        or bool(layout.dictionary_columns)
        for layout in ir.input_layouts
    ):
        interaction.add("layout_sensitive")
    if any(layout.row_count <= 1 for layout in ir.input_layouts) \
        and "union_all" in program_kinds \
        and program_kinds.intersection({"groupby", "aggregate"}):
        interaction.add("empty_union_aggregate")
    if len(relevant_kinds) >= 2 or len(declaration.anchor_node_ids) >= 2:
        interaction.add("multi_operation")

    selected_edges = [
        (left, right)
        for left, right in zip(ir.nodes, ir.nodes[1:])
        if not anchor_ids or left.node_id in anchor_ids or right.node_id in anchor_ids
    ]
    for left, right in selected_edges:
        edge = (left.kind, right.kind)
        if edge[0] == "sort" and edge[1] in {"limit", "offset"}:
            transitions.add("sort_to_limit_or_offset")
        if edge[0] in {"limit", "offset"} and edge[1] in {"groupby", "aggregate"}:
            transitions.add("limit_or_offset_to_aggregate")
        if edge[0] in {"groupby", "aggregate"} and edge[1] == "sort":
            transitions.add("aggregate_to_sort")
        if edge[0] == "filter" and edge[1] in {"join", "semi_join", "anti_join"}:
            transitions.add("filter_to_join")
        if edge[0] in {"join", "semi_join", "anti_join"} and edge[1] in {"filter", "limit", "offset"}:
            transitions.add("join_to_filter_or_limit")
        if edge[0] == "distinct" and edge[1] in {"sort", "limit", "offset", "running_sum"}:
            transitions.add("distinct_to_order")
        if edge[0] == "union_all" and edge[1] == "distinct":
            transitions.add("setop_to_distinct")
        if any(expression.kind == "cast" for expression in left.scalar_expressions) \
            and edge[1] in {"join", "semi_join", "anti_join"}:
            transitions.add("cast_to_membership")
    target_fault_models = set(declaration.obligation.target_fault_models)
    interaction = {
        feature
        for feature in interaction
        if _feature_applies_to_fault_models(
            feature,
            target_fault_models,
            transition=False,
        )
    }
    transitions = {
        feature
        for feature in transitions
        if _feature_applies_to_fault_models(
            feature,
            target_fault_models,
            transition=True,
        )
    }
    return tuple(sorted(interaction)), tuple(sorted(transitions))


def _feature_applies_to_fault_models(
    feature: str,
    fault_models: set[str],
    *,
    transition: bool,
) -> bool:
    if feature == "multi_operation":
        return True
    compatibility = (
        {
            "sort_to_limit_or_offset": {"ordering_stability", "limit_pushdown", "offset_semantics", "null_placement"},
            "limit_or_offset_to_aggregate": {"ordering_stability", "limit_pushdown", "offset_semantics", "aggregation_strategy"},
            "aggregate_to_sort": {"ordering_stability", "null_placement", "aggregation_null_semantics", "aggregation_projection"},
            "filter_to_join": {"join_null_semantics", "join_cardinality", "filter_pushdown", "predicate_semantics"},
            "join_to_filter_or_limit": {"join_cardinality", "filter_pushdown", "limit_pushdown", "join_ordering"},
            "distinct_to_order": {"ordering_stability", "null_placement", "bag_semantics", "duplicate_semantics"},
            "cast_to_membership": {"cast_boundary", "dtype_lowering", "numeric_precision", "join_null_semantics"},
            "setop_to_distinct": {"union_all_semantics", "bag_semantics", "duplicate_semantics"},
        }
        if transition
        else {
            "order_limit_offset": {"ordering_stability", "limit_pushdown", "offset_semantics", "null_placement"},
            "null_aggregation_order": {"aggregation_null_semantics", "null_placement", "ordering_stability"},
            "join_predicate_limit": {"join_cardinality", "join_null_semantics", "filter_pushdown", "limit_pushdown"},
            "cast_membership_window": {"cast_boundary", "dtype_lowering", "numeric_precision", "join_null_semantics"},
            "set_bag_window": {"union_all_semantics", "bag_semantics", "duplicate_semantics", "ordering_stability"},
            "nullable_membership": {"join_null_semantics", "null_semantics", "predicate_semantics"},
            "numeric_window": {"numeric_precision", "ordering_stability", "partitioning"},
            "layout_sensitive": {"input_materialization", "dtype_lowering", "schema_ordering", "partitioning"},
            "empty_union_aggregate": {"empty_input_handling", "union_all_semantics", "aggregation_null_semantics"},
        }
    )
    allowed = compatibility.get(feature, set())
    return bool(allowed.intersection(fault_models))


def _executable_obligation(
    case: Case,
    relation: str,
    declaration: ObligationDeclaration,
    *,
    registered: bool,
    variant_count: int,
    priority: dict[str, Any],
) -> ExecutableObligation:
    obligation = declaration.obligation
    preconditions_satisfied = _preconditions_satisfied(case, obligation.preconditions)
    applicable = registered and preconditions_satisfied and variant_count > 0
    if not registered:
        skip_reason = "transformation_not_registered"
    elif not preconditions_satisfied:
        skip_reason = "precondition_failed"
    elif variant_count <= 0:
        skip_reason = "transformation_not_constructible"
    else:
        skip_reason = ""
    return ExecutableObligation(
        obligation=obligation,
        declaration_scopes=declaration.declaration_scopes,
        anchor_node_ids=declaration.anchor_node_ids,
        registered=registered,
        applicable=applicable,
        variant_count=variant_count,
        priority=priority,
        skip_reason=skip_reason,
    )


def _preconditions_satisfied(case: Case, preconditions: Iterable[str]) -> bool:
    checks = {
        "input_schema_valid": bool(case.tables)
        and all(
            len({column.name for column in table.columns}) == len(table.columns)
            for table in case.tables
        ),
        "secondary_relation_available": len(case.tables) > 1,
        "order_keys_resolvable": True,
        "program_pattern_matched": True,
    }
    return all(checks.get(str(precondition), False) for precondition in preconditions)


def _variant_order_key(relations: list[str]):
    order = {relation: index for index, relation in enumerate(relations)}
    return lambda variant: (order.get(variant.relation, len(order)), variant.name)
