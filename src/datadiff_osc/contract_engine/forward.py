from __future__ import annotations

from dataclasses import dataclass

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.domains import AbstractState, DefinednessDomain
from datadiff_osc.contract_engine.rules import (
    ContractRuleRegistry,
    RuleApplication,
    SemanticStep,
    UnknownSemanticRule,
)


@dataclass(frozen=True, slots=True)
class ForwardAnalysis:
    initial_state: AbstractState
    states: tuple[AbstractState, ...]
    applications: tuple[RuleApplication, ...]
    unresolved_rules: tuple[str, ...]

    @property
    def final_state(self) -> AbstractState:
        return self.states[-1] if self.states else self.initial_state

    @property
    def digest(self) -> str:
        return stable_digest("osc-forward-analysis", self)

    def to_dict(self) -> dict:
        return to_primitive(self)


def forward_analyze(
    initial_state: AbstractState,
    steps: tuple[SemanticStep, ...],
    registry: ContractRuleRegistry,
) -> ForwardAnalysis:
    state = initial_state
    states: list[AbstractState] = []
    applications: list[RuleApplication] = []
    unresolved: list[str] = []
    for step in steps:
        subjects = (
            (("operation", step.kind),)
            + tuple(("expression", item) for item in step.expression_kinds)
            + tuple(("aggregate", item) for item in step.aggregate_kinds)
        )
        for category, subject in subjects:
            try:
                transformer = registry.resolve(category, subject)
            except UnknownSemanticRule:
                unresolved.append(f"{category}:{subject}")
                state = state.evolve(definedness=DefinednessDomain.UNKNOWN)
                continue
            state, application = transformer.forward(state, step)
            applications.append(application)
        states.append(state)
    return ForwardAnalysis(
        initial_state=initial_state,
        states=tuple(states),
        applications=tuple(applications),
        unresolved_rules=tuple(dict.fromkeys(unresolved)),
    )

