from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from datadiff_osc._canonical import frozen_pairs, pairs_dict, stable_digest, to_primitive
from datadiff_osc.contract_engine.domains import (
    AbstractState,
    CardinalityDomain,
    DefinednessDomain,
    DeterminismDomain,
    MultiplicityDomain,
    NullDomain,
    ObservationDemand,
    OrderDomain,
    PartitionOrderDomain,
    SchemaKnowledge,
    SpecialFloatDomain,
    TieDeterminismDomain,
)


OPERATION_KINDS: tuple[str, ...] = (
    "aggregate", "anti_join", "case_when", "coalesce", "distinct",
    "drop_nulls", "fill_null", "filter", "groupby", "join", "limit",
    "mutate", "offset", "row_number_filter", "running_sum", "select",
    "semi_join", "sort", "sortedness_check", "tuple_absence_filter",
    "union_all",
)

EXPRESSION_KINDS: tuple[str, ...] = (
    "abs", "add_const", "arith_const", "bool_not", "cast", "clip",
    "date_part", "reverse_division_columns", "string_basename",
    "string_concat", "string_contains", "string_ends_with",
    "string_length", "string_lower", "string_null_if_empty",
    "string_replace", "string_slice", "string_split_part",
    "string_starts_with", "string_strip", "string_upper",
)

AGGREGATE_KINDS: tuple[str, ...] = (
    "all", "any", "count", "max", "mean", "min", "nunique", "sum",
)


class UnknownSemanticRule(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class SemanticStep:
    step_id: str
    kind: str
    arguments: tuple[tuple[str, Any], ...] = ()
    expression_kinds: tuple[str, ...] = ()
    aggregate_kinds: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        step_id: str,
        kind: str,
        arguments: dict[str, Any] | None = None,
        expression_kinds: tuple[str, ...] = (),
        aggregate_kinds: tuple[str, ...] = (),
    ) -> "SemanticStep":
        return cls(
            step_id=step_id,
            kind=kind,
            arguments=frozen_pairs(arguments),
            expression_kinds=tuple(expression_kinds),
            aggregate_kinds=tuple(aggregate_kinds),
        )

    @property
    def args(self) -> dict[str, Any]:
        return pairs_dict(self.arguments)


@dataclass(frozen=True, slots=True)
class RuleApplication:
    rule_id: str
    category: str
    subject: str
    input_digest: str
    output_digest: str
    preconditions: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()
    facts: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


ForwardRule = Callable[[AbstractState, SemanticStep, str], tuple[AbstractState, tuple[str, ...], tuple[str, ...], dict[str, Any]]]
BackwardRule = Callable[
    [ObservationDemand, SemanticStep, str, AbstractState], ObservationDemand
]


@dataclass(frozen=True, slots=True)
class ContractTransformer:
    rule_id: str
    category: str
    subject: str
    forward_rule: ForwardRule
    backward_rule: BackwardRule

    def forward(
        self, state: AbstractState, step: SemanticStep
    ) -> tuple[AbstractState, RuleApplication]:
        output, preconditions, obligations, facts = self.forward_rule(
            state, step, self.subject
        )
        return output, RuleApplication(
            rule_id=self.rule_id,
            category=self.category,
            subject=self.subject,
            input_digest=state.digest,
            output_digest=output.digest,
            preconditions=preconditions,
            obligations=obligations,
            facts=frozen_pairs(facts),
        )

    def backward(
        self,
        demand: ObservationDemand,
        step: SemanticStep,
        forward_state: AbstractState | None = None,
    ) -> ObservationDemand:
        state = forward_state or AbstractState()
        return self.backward_rule(demand, step, self.subject, state)


@dataclass(frozen=True, slots=True)
class ContractRuleRegistry:
    operation: tuple[ContractTransformer, ...]
    expression: tuple[ContractTransformer, ...]
    aggregate: tuple[ContractTransformer, ...]
    schema_version: str = "osc-contract-rule-registry-v1"

    @property
    def digest(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "rules": [
                {"id": rule.rule_id, "category": rule.category, "subject": rule.subject}
                for rule in (*self.operation, *self.expression, *self.aggregate)
            ],
        }
        return stable_digest("osc-rule-registry", payload)

    def resolve(self, category: str, subject: str) -> ContractTransformer:
        rules = getattr(self, category, ())
        match = next((item for item in rules if item.subject == subject), None)
        if match is None:
            raise UnknownSemanticRule(f"unknown {category} semantic rule: {subject}")
        return match

    def coverage(self) -> dict[str, tuple[str, ...]]:
        return {
            "operation": tuple(item.subject for item in self.operation),
            "expression": tuple(item.subject for item in self.expression),
            "aggregate": tuple(item.subject for item in self.aggregate),
        }


def _operation_forward(
    state: AbstractState,
    step: SemanticStep,
    kind: str,
) -> tuple[AbstractState, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    args = step.args
    preconditions = ["input_schema_valid"]
    obligations: list[str] = []
    changes: dict[str, Any] = {}
    facts: dict[str, Any] = {"operation": kind}

    if kind in {"filter", "drop_nulls", "tuple_absence_filter"}:
        changes["cardinality"] = CardinalityDomain.UNKNOWN
        if kind == "drop_nulls" and bool(args.get("all_columns", False)):
            changes["nulls"] = NullDomain.NONE
        if kind == "tuple_absence_filter":
            preconditions.append("secondary_relation_available")
            obligations.append("nullable_tuple_membership")
    elif kind in {"fill_null", "coalesce"}:
        changes["nulls"] = NullDomain.NONE if bool(args.get("replacement_non_null", True)) else NullDomain.MAYBE
        obligations.append("null_rewrite_exact")
    elif kind == "case_when":
        changes["nulls"] = NullDomain.MAYBE
        obligations.append("predicate_truth_table")
    elif kind == "distinct":
        changes.update(
            multiplicity=MultiplicityDomain.SET_OUTPUT,
            presentation_order=OrderDomain.NONE,
            evaluation_order=OrderDomain.NONE,
            tie_determinism=TieDeterminismDomain.NOT_APPLICABLE,
        )
        obligations.extend(("set_membership", "output_uniqueness", "exact_cardinality"))
    elif kind in {"groupby", "aggregate"}:
        changes.update(
            cardinality=CardinalityDomain.UNKNOWN,
            multiplicity=MultiplicityDomain.UNIQUE if kind == "aggregate" else MultiplicityDomain.SET_OUTPUT,
            evaluation_order=OrderDomain.NONE,
            presentation_order=OrderDomain.NONE,
            tie_determinism=TieDeterminismDomain.NOT_APPLICABLE,
            partition_order=PartitionOrderDomain.UNORDERED,
        )
        obligations.extend(("group_membership", "aggregate_component_relations"))
    elif kind in {"join", "semi_join", "anti_join"}:
        preconditions.append("secondary_relation_available")
        changes.update(
            cardinality=CardinalityDomain.UNKNOWN,
            evaluation_order=OrderDomain.UNKNOWN,
            presentation_order=OrderDomain.NONE,
            tie_determinism=TieDeterminismDomain.NOT_APPLICABLE,
        )
        if kind == "join":
            changes["multiplicity"] = MultiplicityDomain.MAY_DUPLICATE
            obligations.append("join_product_multiplicity")
        else:
            obligations.extend(("left_multiplicity", "right_duplicate_invariance"))
        obligations.append("nullable_key_membership")
    elif kind == "sort":
        preconditions.append("order_keys_resolvable")
        total = bool(args.get("total_order", False))
        changes.update(
            evaluation_order=OrderDomain.TOTAL if total else OrderDomain.PARTIAL,
            presentation_order=OrderDomain.TOTAL if total else OrderDomain.PARTIAL,
            tie_determinism=(TieDeterminismDomain.DETERMINISTIC if total else TieDeterminismDomain.NONDETERMINISTIC),
        )
        obligations.append("exact_sort_keys")
        if not total:
            obligations.append("partial_order_tie_freedom")
    elif kind in {"limit", "offset"}:
        changes["cardinality"] = CardinalityDomain.UNKNOWN
        obligations.extend(("evaluation_order_required", "bounded_cardinality"))
        if state.evaluation_order in {OrderDomain.NONE, OrderDomain.UNKNOWN}:
            changes["definedness"] = DefinednessDomain.UNKNOWN
            facts["unresolved_order"] = True
    elif kind in {"running_sum", "row_number_filter"}:
        preconditions.append("order_keys_resolvable")
        total = bool(args.get("total_order", False))
        partition_keys = tuple(str(item) for item in args.get("partition_by", ()) or ())
        changes.update(
            evaluation_order=OrderDomain.TOTAL if total else OrderDomain.PARTIAL,
            presentation_order=OrderDomain.TOTAL if total else OrderDomain.PARTIAL,
            tie_determinism=(TieDeterminismDomain.DETERMINISTIC if total else TieDeterminismDomain.NONDETERMINISTIC),
            partition_order=PartitionOrderDomain.ORDERED,
            partition_keys=partition_keys,
        )
        obligations.extend(("evaluation_order_required", "partition_order_required"))
        if kind == "running_sum":
            obligations.append("numeric_accumulation")
    elif kind == "sortedness_check":
        obligations.extend(("evaluation_order_required", "sortedness_predicate"))
    elif kind == "union_all":
        preconditions.append("secondary_relation_available")
        changes.update(
            cardinality=CardinalityDomain.UNKNOWN,
            multiplicity=MultiplicityDomain.MAY_DUPLICATE,
            presentation_order=OrderDomain.NONE,
            evaluation_order=OrderDomain.UNKNOWN,
        )
        obligations.append("duplicate_multiplicity")
    elif kind in {"select", "mutate"}:
        pass
    else:  # registry construction prevents this; retain a defensive fail closed.
        changes.update(
            schema_knowledge=SchemaKnowledge.UNKNOWN,
            definedness=DefinednessDomain.UNKNOWN,
            evaluation_order=OrderDomain.UNKNOWN,
            presentation_order=OrderDomain.UNKNOWN,
            determinism=DeterminismDomain.UNKNOWN,
        )
        obligations.append("unknown_operation")

    return state.evolve(**changes), tuple(preconditions), tuple(obligations), facts


def _operation_backward(
    demand: ObservationDemand,
    step: SemanticStep,
    kind: str,
    forward_state: AbstractState,
) -> ObservationDemand:
    changes: dict[str, Any] = {}
    if kind in {"limit", "offset"}:
        changes["evaluation_order"] = True
        changes["tie_determinism"] = True
        if forward_state.definedness == DefinednessDomain.UNKNOWN:
            changes["determinism"] = True
    elif kind in {"running_sum", "row_number_filter"}:
        changes.update(evaluation_order=True, partition_order=True, tie_determinism=True)
        if kind == "running_sum":
            changes["numeric"] = True
    elif kind == "sort":
        # Sort discharges upstream presentation order, but its keys/NULL policy
        # remain demanded as evaluation facts.
        if forward_state.presentation_order in {OrderDomain.PARTIAL, OrderDomain.TOTAL}:
            changes["presentation_order"] = False
        else:
            changes["presentation_order"] = True
            changes["tie_determinism"] = True
        changes.update(evaluation_order=False, null_semantics=True)
    elif kind == "distinct":
        changes.update(
            duplicate_multiplicity=(
                False
                if forward_state.multiplicity == MultiplicityDomain.SET_OUTPUT
                else demand.duplicate_multiplicity
            ),
            row_membership=True,
        )
    elif kind in {"groupby", "aggregate"}:
        changes.update(presentation_order=False, null_semantics=True)
    elif kind in {"join", "semi_join", "anti_join", "tuple_absence_filter"}:
        changes.update(presentation_order=False, null_semantics=True)
    elif kind in {"fill_null", "coalesce", "case_when", "drop_nulls", "filter"}:
        changes["null_semantics"] = True
    return demand.evolve(**changes)


def _expression_forward(
    state: AbstractState,
    step: SemanticStep,
    kind: str,
) -> tuple[AbstractState, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    args = step.args
    preconditions = ("expression_inputs_typed",)
    obligations: list[str] = [f"expression:{kind}:value"]
    changes: dict[str, Any] = {}
    if kind in {"cast", "date_part"}:
        obligations.extend(("logical_dtype_exact", "accept_reject_exact"))
        if not bool(args.get("domain_proven", False)):
            changes["definedness"] = DefinednessDomain.UNKNOWN
    if kind in {"arith_const", "reverse_division_columns"}:
        obligations.extend(("numeric_role_relation", "domain_error_relation"))
        changes["special_floats"] = SpecialFloatDomain.MAYBE
    elif kind in {"abs", "add_const", "clip"}:
        obligations.append("numeric_role_relation")
    if kind in {"bool_not", "string_null_if_empty"}:
        obligations.append("nullable_scalar_truth")
    if kind.startswith("string_"):
        obligations.append("unicode_scalar_semantics")
    return state.evolve(**changes), preconditions, tuple(obligations), {"expression": kind}


def _expression_backward(
    demand: ObservationDemand,
    step: SemanticStep,
    kind: str,
    forward_state: AbstractState,
) -> ObservationDemand:
    changes: dict[str, Any] = {}
    if kind in {"cast", "date_part"}:
        changes.update(schema_types=True, nullability=True, error_category=True)
        if forward_state.definedness == DefinednessDomain.UNKNOWN:
            changes["status"] = True
    if kind in {"abs", "add_const", "arith_const", "clip", "reverse_division_columns"}:
        changes["numeric"] = True
    if kind in {"bool_not", "string_null_if_empty"}:
        changes["null_semantics"] = True
    return demand.evolve(**changes)


def _aggregate_forward(
    state: AbstractState,
    step: SemanticStep,
    kind: str,
) -> tuple[AbstractState, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    obligations = [f"aggregate:{kind}:null_policy", f"aggregate:{kind}:dtype"]
    changes: dict[str, Any] = {"cardinality": CardinalityDomain.UNKNOWN}
    if kind in {"sum", "mean", "min", "max"}:
        obligations.append(f"aggregate:{kind}:special_float")
    if kind in {"any", "all"}:
        obligations.append(f"aggregate:{kind}:three_valued_boolean")
    if kind == "count":
        changes["nulls"] = NullDomain.NONE
    else:
        changes["nulls"] = NullDomain.MAYBE
    return state.evolve(**changes), ("aggregate_input_typed",), tuple(obligations), {"aggregate": kind}


def _aggregate_backward(
    demand: ObservationDemand,
    step: SemanticStep,
    kind: str,
    forward_state: AbstractState,
) -> ObservationDemand:
    changes: dict[str, Any] = {"null_semantics": True}
    if kind in {"sum", "mean", "min", "max"}:
        changes["numeric"] = True
    return demand.evolve(**changes)


def default_rule_registry() -> ContractRuleRegistry:
    return ContractRuleRegistry(
        operation=tuple(
            ContractTransformer(
                rule_id=f"osc.op.{kind}.v1",
                category="operation",
                subject=kind,
                forward_rule=_operation_forward,
                backward_rule=_operation_backward,
            )
            for kind in OPERATION_KINDS
        ),
        expression=tuple(
            ContractTransformer(
                rule_id=f"osc.expr.{kind}.v1",
                category="expression",
                subject=kind,
                forward_rule=_expression_forward,
                backward_rule=_expression_backward,
            )
            for kind in EXPRESSION_KINDS
        ),
        aggregate=tuple(
            ContractTransformer(
                rule_id=f"osc.agg.{kind}.v1",
                category="aggregate",
                subject=kind,
                forward_rule=_aggregate_forward,
                backward_rule=_aggregate_backward,
            )
            for kind in AGGREGATE_KINDS
        ),
    )
