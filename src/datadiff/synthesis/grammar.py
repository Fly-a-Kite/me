from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence

from datadiff.dsl import TableData
from datadiff.synthesis.typed_state import TypedProgramState


class RuleContext(Protocol):
    primary_table: TableData
    extra_tables: Sequence[TableData]
    table_by_name: Mapping[str, TableData]


RulePrecondition = Callable[[TypedProgramState], bool]
RuleGenerator = Callable[[random.Random, TypedProgramState, RuleContext], dict[str, Any]]
RuleTransition = Callable[[TypedProgramState, Mapping[str, Any], RuleContext], TypedProgramState]


@dataclass(frozen=True, slots=True)
class ProductionRule:
    op_kind: str
    weight: float
    precondition: RulePrecondition
    generator: RuleGenerator
    state_transition: RuleTransition


class GrammarRegistry:
    def __init__(self, rules: Sequence[ProductionRule] | None = None) -> None:
        self._rules: dict[str, ProductionRule] = {}
        for rule in rules or ():
            self.register(rule)

    def register(self, rule: ProductionRule) -> None:
        self._rules[str(rule.op_kind)] = rule

    def available_productions(self, state: TypedProgramState) -> list[ProductionRule]:
        return [rule for rule in self._rules.values() if rule.precondition(state)]

    def synthesize_step(
        self,
        rnd: random.Random,
        state: TypedProgramState,
        context: RuleContext,
        *,
        weights: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, Any], TypedProgramState]:
        available = self.available_productions(state)
        if not available:
            raise ValueError("no typed grammar productions available")
        if weights:
            available = [
                replace(rule, weight=max(0.0, rule.weight * float(weights.get(rule.op_kind, 1.0))))
                for rule in available
            ]
        selected = _weighted_choice(rnd, available)
        operation = selected.generator(rnd, state, context)
        return operation, selected.state_transition(state, operation, context)

    def to_summary(self) -> dict[str, Any]:
        return {
            "rule_count": len(self._rules),
            "rules": [
                {"op_kind": rule.op_kind, "weight": rule.weight}
                for rule in sorted(self._rules.values(), key=lambda item: item.op_kind)
            ],
        }


def transition_by_program_state(
    state: TypedProgramState,
    operation: Mapping[str, Any],
    context: RuleContext,
) -> TypedProgramState:
    return state.after_operation(operation, tables=context.table_by_name)


def _weighted_choice(rnd: random.Random, rules: Sequence[ProductionRule]) -> ProductionRule:
    total = sum(max(0.0, float(rule.weight)) for rule in rules)
    if total <= 0.0:
        return rnd.choice(list(rules))
    threshold = rnd.random() * total
    cumulative = 0.0
    for rule in rules:
        cumulative += max(0.0, float(rule.weight))
        if threshold <= cumulative:
            return rule
    return rules[-1]
