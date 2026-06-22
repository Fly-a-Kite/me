from __future__ import annotations

from dataclasses import dataclass


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


def bounded_ratio(numerator: float, denominator: float, *, upper: float = 1.0) -> float:
    if denominator <= 0.0:
        return 0.0
    return _clamp(float(numerator) / float(denominator), lower=0.0, upper=float(upper))


def novelty_decay(count: float, *, scale: float = 1.0, floor: float = 0.0) -> float:
    score = float(scale) / (1.0 + max(0.0, float(count)))
    return max(float(floor), score)


@dataclass(frozen=True, slots=True)
class ObjectiveVector:
    discovery: float = 0.0
    semantic: float = 0.0
    novelty: float = 0.0
    expandability: float = 0.0
    structural_risk: float = 0.0
    coverage_gain: float = 0.0


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    discovery: float
    semantic: float
    novelty: float
    expandability: float
    structural_risk: float
    coverage_gain: float


@dataclass(frozen=True, slots=True)
class CostVector:
    invalidity: float = 0.0
    false_positive: float = 0.0
    redundancy: float = 0.0
    target_miss: float = 0.0
    runtime_cost: float = 0.0


@dataclass(frozen=True, slots=True)
class CostWeights:
    invalid: float = 0.0
    false_positive: float = 0.0
    redundant: float = 0.0
    miss: float = 0.0
    runtime: float = 0.0


@dataclass(frozen=True, slots=True)
class ConstraintThresholds:
    min_validity: float = 0.0
    max_false_positive_risk: float = 1.0
    reject_modifier: float = 0.05
    strong_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class MultiObjectiveSpec:
    objective: ObjectiveWeights
    cost: CostWeights
    constraints: ConstraintThresholds = ConstraintThresholds()
    objective_scale: float = 1.0
    additive_bias: float = 0.0


def weighted_objective_value(
    vector: ObjectiveVector,
    *,
    weights: ObjectiveWeights,
) -> float:
    return (
        float(weights.discovery) * float(vector.discovery)
        + float(weights.semantic) * float(vector.semantic)
        + float(weights.novelty) * float(vector.novelty)
        + float(weights.expandability) * float(vector.expandability)
        + float(weights.structural_risk) * float(vector.structural_risk)
        + float(weights.coverage_gain) * float(vector.coverage_gain)
    )


def additive_cost_penalty(
    vector: CostVector,
    *,
    weights: CostWeights,
) -> float:
    return (
        float(weights.invalid) * max(0.0, float(vector.invalidity))
        + float(weights.false_positive) * max(0.0, float(vector.false_positive))
        + float(weights.redundant) * max(0.0, float(vector.redundancy))
        + float(weights.miss) * max(0.0, float(vector.target_miss))
        + float(weights.runtime) * max(0.0, float(vector.runtime_cost))
    )


def multiplicative_cost_modifier(
    vector: CostVector,
    *,
    weights: CostWeights,
    min_modifier: float = 0.05,
    max_modifier: float = 1.20,
) -> float:
    penalty = additive_cost_penalty(vector, weights=weights)
    return _clamp(1.0 - penalty, lower=float(min_modifier), upper=float(max_modifier))


def constrained_objective_score(
    vector: ObjectiveVector,
    *,
    cost: CostVector,
    spec: MultiObjectiveSpec,
    validity: float = 1.0,
    false_positive_risk: float = 0.0,
) -> float:
    objective_value = weighted_objective_value(vector, weights=spec.objective)
    penalty = additive_cost_penalty(cost, weights=spec.cost)
    score = (float(spec.objective_scale) * objective_value) - penalty + float(spec.additive_bias)
    thresholds = spec.constraints
    if float(validity) < float(thresholds.min_validity):
        return min(score, float(thresholds.reject_modifier) * score)
    if float(false_positive_risk) > float(thresholds.max_false_positive_risk):
        score -= float(thresholds.strong_penalty)
    return score


SEED_OBJECTIVE_SPEC = MultiObjectiveSpec(
    objective=ObjectiveWeights(
        discovery=0.50,
        semantic=0.22,
        novelty=0.06,
        expandability=0.08,
        structural_risk=0.06,
        coverage_gain=0.08,
    ),
    cost=CostWeights(
        invalid=0.18,
        false_positive=0.90,
        redundant=0.16,
        miss=0.0,
        runtime=0.10,
    ),
    constraints=ConstraintThresholds(
        min_validity=0.55,
        max_false_positive_risk=0.70,
        reject_modifier=0.15,
        strong_penalty=0.45,
    ),
    objective_scale=3.0,
)

SOURCE_OBJECTIVE_SPEC = MultiObjectiveSpec(
    objective=ObjectiveWeights(
        discovery=1.05,
        semantic=0.16,
        novelty=0.14,
        expandability=0.08,
        structural_risk=0.06,
        coverage_gain=0.08,
    ),
    cost=CostWeights(
        invalid=0.55,
        false_positive=1.55,
        redundant=0.18,
        miss=0.0,
        runtime=0.0,
    ),
    constraints=ConstraintThresholds(
        min_validity=0.65,
        max_false_positive_risk=0.55,
        reject_modifier=0.10,
        strong_penalty=0.85,
    ),
    objective_scale=2.15,
)

OPERATOR_OBJECTIVE_SPEC = MultiObjectiveSpec(
    objective=ObjectiveWeights(
        discovery=0.72,
        semantic=0.34,
        novelty=0.10,
        expandability=0.18,
        structural_risk=0.28,
        coverage_gain=0.12,
    ),
    cost=CostWeights(
        invalid=0.0,
        false_positive=0.0,
        redundant=0.35,
        miss=0.0,
        runtime=0.0,
    ),
    constraints=ConstraintThresholds(
        min_validity=0.0,
        max_false_positive_risk=1.0,
        reject_modifier=1.0,
        strong_penalty=0.0,
    ),
)

SEED_FRONTIER_OBJECTIVE_SPEC = MultiObjectiveSpec(
    objective=ObjectiveWeights(
        discovery=0.34,
        semantic=0.20,
        novelty=0.16,
        expandability=0.14,
        structural_risk=0.08,
        coverage_gain=0.08,
    ),
    cost=CostWeights(
        invalid=0.18,
        false_positive=0.0,
        redundant=0.30,
        miss=0.12,
        runtime=0.0,
    ),
    constraints=ConstraintThresholds(
        min_validity=0.45,
        max_false_positive_risk=1.0,
        reject_modifier=0.20,
        strong_penalty=0.0,
    ),
)

PLAN_STEP_OBJECTIVE_SPEC = MultiObjectiveSpec(
    objective=ObjectiveWeights(
        discovery=0.32,
        semantic=0.26,
        novelty=0.10,
        expandability=0.12,
        structural_risk=0.12,
        coverage_gain=0.08,
    ),
    cost=CostWeights(
        invalid=0.50,
        false_positive=0.0,
        redundant=0.18,
        miss=0.0,
        runtime=0.10,
    ),
    constraints=ConstraintThresholds(
        min_validity=0.55,
        max_false_positive_risk=1.0,
        reject_modifier=0.10,
        strong_penalty=0.0,
    ),
)

BATCH_OBJECTIVE_SPEC = MultiObjectiveSpec(
    objective=ObjectiveWeights(
        discovery=0.90,
        semantic=0.55,
        novelty=0.12,
        expandability=0.08,
        structural_risk=0.10,
        coverage_gain=0.20,
    ),
    cost=CostWeights(
        invalid=0.90,
        false_positive=1.80,
        redundant=0.45,
        miss=0.70,
        runtime=0.35,
    ),
    constraints=ConstraintThresholds(
        min_validity=0.50,
        max_false_positive_risk=0.70,
        reject_modifier=0.10,
        strong_penalty=0.90,
    ),
)
