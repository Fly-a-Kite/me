from __future__ import annotations

from dataclasses import dataclass

from datadiff_osc._canonical import stable_digest, to_primitive
from datadiff_osc.contract_engine.domains import ObservationDemand
from datadiff_osc.contract_engine.domains import AbstractState, DefinednessDomain
from datadiff_osc.contract_engine.rules import (
    ContractRuleRegistry,
    SemanticStep,
    UnknownSemanticRule,
)


@dataclass(frozen=True, slots=True)
class DemandTrace:
    step_id: str
    rule_ids: tuple[str, ...]
    downstream: ObservationDemand
    upstream: ObservationDemand
    forward_state_digest: str = ""
    fail_closed: bool = False


@dataclass(frozen=True, slots=True)
class BackwardAnalysis:
    final_demand: ObservationDemand
    traces: tuple[DemandTrace, ...]
    unresolved_rules: tuple[str, ...]

    @property
    def input_demand(self) -> ObservationDemand:
        return self.traces[-1].upstream if self.traces else self.final_demand

    @property
    def digest(self) -> str:
        return stable_digest("osc-backward-analysis", self)

    def to_dict(self) -> dict:
        return to_primitive(self)


def demand_for_relation(relation_id: str) -> ObservationDemand:
    if relation_id in {"status_ok", "status_equal"}:
        return ObservationDemand(
            status=True,
            schema_names=False,
            cardinality=False,
            row_membership=False,
            duplicate_multiplicity=False,
        )
    if relation_id in {"sequence_equal", "layout_sequence_equal", "mode_sequence_equal"}:
        return ObservationDemand(
            schema_types=True,
            nullability=True,
            presentation_order=True,
            numeric=True,
            null_semantics=True,
        )
    if relation_id in {"bag_equal", "layout_bag_equal", "mode_bag_equal"}:
        return ObservationDemand(
            schema_types=True,
            nullability=True,
            presentation_order=False,
            numeric=True,
            null_semantics=True,
        )
    if relation_id == "set_equal_unique":
        return ObservationDemand(
            schema_types=True,
            nullability=True,
            presentation_order=False,
            duplicate_multiplicity=True,
            null_semantics=True,
        )
    if relation_id == "containment":
        return ObservationDemand(cardinality=False)
    if relation_id in {"error_category_equal", "accept_reject_equal"}:
        return ObservationDemand(
            schema_names=False,
            cardinality=False,
            row_membership=False,
            duplicate_multiplicity=False,
            error_category=True,
        )
    if relation_id in {"partial_order_equal", "partial_order_topk"}:
        return ObservationDemand(
            schema_types=True,
            nullability=True,
            presentation_order=True,
            tie_determinism=True,
            null_semantics=True,
        )
    if relation_id.startswith("numeric_"):
        return ObservationDemand(numeric=True, schema_types=True, nullability=True)
    return ObservationDemand()


def backward_analyze(
    final_demand: ObservationDemand,
    steps: tuple[SemanticStep, ...],
    registry: ContractRuleRegistry,
    forward_states: tuple[AbstractState, ...] = (),
) -> BackwardAnalysis:
    demand = final_demand
    traces: list[DemandTrace] = []
    unresolved: list[str] = []
    aligned = len(forward_states) == len(steps)
    if forward_states and not aligned:
        unresolved.append("forward_state_alignment")
    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        forward_state = (
            forward_states[index]
            if aligned and forward_states
            else AbstractState()
        )
        downstream = demand
        rule_ids: list[str] = []
        fail_closed = False
        subjects = (
            tuple(("aggregate", item) for item in reversed(step.aggregate_kinds))
            + tuple(("expression", item) for item in reversed(step.expression_kinds))
            + (("operation", step.kind),)
        )
        for category, subject in subjects:
            try:
                transformer = registry.resolve(category, subject)
            except UnknownSemanticRule:
                unresolved.append(f"{category}:{subject}")
                fail_closed = True
                # Unknown behavior cannot discharge any demand.
                demand = demand.evolve(
                    evaluation_order=True,
                    partition_order=True,
                    tie_determinism=True,
                    error_category=True,
                )
                continue
            demand = transformer.backward(demand, step, forward_state)
            rule_ids.append(transformer.rule_id)
        if forward_state.definedness == DefinednessDomain.UNKNOWN:
            demand = demand.evolve(status=True, error_category=True)
            fail_closed = True
        traces.append(
            DemandTrace(
                step_id=step.step_id,
                rule_ids=tuple(rule_ids),
                downstream=downstream,
                upstream=demand,
                forward_state_digest=forward_state.digest,
                fail_closed=fail_closed,
            )
        )
    return BackwardAnalysis(
        final_demand=final_demand,
        traces=tuple(traces),
        unresolved_rules=tuple(dict.fromkeys(unresolved)),
    )
