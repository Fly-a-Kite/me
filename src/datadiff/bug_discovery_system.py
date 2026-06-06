from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, fields
from typing import Any, Mapping

from datadiff.finding_outcomes import (
    candidate_issue_family_keys,
    has_source_issue,
    is_false_positive_finding,
    is_issue_replay_finding,
    is_rewardable_candidate_issue_finding,
)


BUG_DISCOVERY_SYSTEM_SCHEMA_VERSION = "bug-discovery-system-v1"


@dataclass(frozen=True, slots=True)
class BugDiscoveryAcquisitionWeights:
    yield_rate: float = 1.0
    novelty_rate: float = 0.75
    false_positive_penalty: float = 1.25
    uncertainty_weight: float = 0.45
    proof_weight: float = 0.60
    entropy_weight: float = 0.20
    saturation_penalty: float = 0.75
    issue_inspired_weight: float = 0.20
    boltzmann_temperature: float = 0.85

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "BugDiscoveryAcquisitionWeights":
        defaults = cls()
        values = {}
        mapping = data or {}
        for field in fields(cls):
            raw = mapping.get(field.name, getattr(defaults, field.name))
            values[field.name] = _nonnegative_float(raw, getattr(defaults, field.name))
        return cls(**values)


def bug_discovery_system_descriptor() -> dict[str, Any]:
    return {
        "schema_version": BUG_DISCOVERY_SYSTEM_SCHEMA_VERSION,
        "objective": "maximize externally reproducible fresh implementation bug yield per unit budget",
        "stages": [
            "generate_or_mutate_candidate",
            "score_by_acquisition",
            "execute_differential_oracles",
            "filter_rewardable_fresh_candidates",
            "recheck_reproducibility",
            "reduce_preserving_root_cause",
            "write_reproducer_and_issue_draft",
        ],
        "theory": {
            "novelty": "Good-Turing unseen-family probability",
            "exploration": "UCB optimism under sparse lane samples",
            "budget": "Boltzmann allocation over acquisition free energy",
            "candidate_priority": "multi-objective proof/reproducibility/dedup utility",
        },
    }


def good_turing_unseen_probability(family_counts: Mapping[str, int] | Counter[str]) -> float:
    counts = _counter_from_mapping(family_counts)
    total = sum(counts.values())
    if total <= 0:
        return 1.0
    singletons = sum(1 for value in counts.values() if value == 1)
    return min(1.0, max(0.0, singletons / float(total)))


def normalized_entropy(family_counts: Mapping[str, int] | Counter[str]) -> float:
    counts = _counter_from_mapping(family_counts)
    total = sum(counts.values())
    positive = [value for value in counts.values() if value > 0]
    if total <= 0 or len(positive) <= 1:
        return 0.0
    entropy = 0.0
    for count in positive:
        p = count / float(total)
        entropy -= p * math.log(p)
    return min(1.0, max(0.0, entropy / math.log(len(positive))))


def lane_true_bug_acquisition(
    lane_metrics: Mapping[str, Any],
    *,
    total_completed_runs: int,
    weights: Mapping[str, Any] | BugDiscoveryAcquisitionWeights | None = None,
) -> dict[str, float]:
    resolved_weights = (
        weights
        if isinstance(weights, BugDiscoveryAcquisitionWeights)
        else BugDiscoveryAcquisitionWeights.from_mapping(weights)
    )
    completed_runs = _nonnegative_int(lane_metrics.get("completed_runs", 0), 0)
    fresh_total = _nonnegative_int(lane_metrics.get("fresh_candidate_total", 0), 0)
    issue_inspired_total = _nonnegative_int(lane_metrics.get("issue_inspired_total", 0), 0)
    known_total = _nonnegative_int(lane_metrics.get("known_saturated_total", 0), 0)
    candidate_total = _nonnegative_int(lane_metrics.get("candidate_total", 0), 0)
    false_positive_total = _nonnegative_int(lane_metrics.get("false_positive_total", 0), 0)
    first_seen_count = _nonnegative_int(lane_metrics.get("first_seen_family_count", 0), 0)
    unique_fresh_count = _nonnegative_int(lane_metrics.get("unique_fresh_family_count", 0), 0)
    family_counts = _counter_from_mapping(lane_metrics.get("fresh_family_counts", {}))

    yield_rate = min(3.0, fresh_total / completed_runs) if completed_runs else 0.0
    novelty_rate = min(1.0, first_seen_count / unique_fresh_count) if unique_fresh_count else 0.0
    false_positive_rate = (
        false_positive_total / (candidate_total + false_positive_total)
        if (candidate_total + false_positive_total) > 0
        else 0.0
    )
    candidate_signal_total = max(candidate_total, fresh_total + issue_inspired_total + known_total)
    saturation_rate = known_total / candidate_signal_total if candidate_signal_total > 0 else 0.0
    issue_inspired_rate = issue_inspired_total / completed_runs if completed_runs else 0.0
    unseen_probability = good_turing_unseen_probability(family_counts)
    entropy = normalized_entropy(family_counts)
    ucb_bonus = resolved_weights.uncertainty_weight * math.sqrt(
        (2.0 * math.log(max(2, _nonnegative_int(total_completed_runs, 0) + 1))) / (completed_runs + 1.0)
    )
    proof_yield = min(1.0, fresh_total / max(1.0, candidate_total + false_positive_total))
    exploitation = (
        1.0
        + resolved_weights.yield_rate * yield_rate
        + resolved_weights.novelty_rate * max(novelty_rate, unseen_probability)
        + resolved_weights.issue_inspired_weight * issue_inspired_rate
        + resolved_weights.proof_weight * proof_yield
        - resolved_weights.false_positive_penalty * false_positive_rate
        - resolved_weights.saturation_penalty * saturation_rate
    )
    free_energy = (
        -exploitation
        + resolved_weights.false_positive_penalty * false_positive_rate
        + resolved_weights.saturation_penalty * saturation_rate
        - resolved_weights.entropy_weight * entropy
    )
    acquisition_score = max(
        0.05,
        exploitation + ucb_bonus + resolved_weights.entropy_weight * entropy,
    )
    return {
        "acquisition_score": acquisition_score,
        "yield_rate": yield_rate,
        "novelty_rate": novelty_rate,
        "good_turing_unseen_probability": unseen_probability,
        "ucb_bonus": ucb_bonus,
        "family_entropy": entropy,
        "false_positive_rate": false_positive_rate,
        "saturation_rate": saturation_rate,
        "issue_inspired_rate": issue_inspired_rate,
        "proof_yield": proof_yield,
        "free_energy": free_energy,
    }


def boltzmann_budget_multipliers(
    scores: list[float] | tuple[float, ...],
    *,
    temperature: float = 0.85,
    min_multiplier: float = 0.5,
    max_multiplier: float = 2.5,
) -> list[float]:
    if not scores:
        return []
    sanitized_scores = [_nonnegative_float(score, 0.0) for score in scores]
    temp = max(1e-6, _nonnegative_float(temperature, 0.85))
    max_score = max(sanitized_scores)
    weights = [
        math.exp(max(-60.0, min(60.0, (score - max_score) / temp)))
        for score in sanitized_scores
    ]
    lower = _nonnegative_float(min_multiplier, 0.5)
    upper = max(lower, _nonnegative_float(max_multiplier, 2.5))
    return _bounded_boltzmann_allocation(
        weights,
        min_multiplier=lower,
        max_multiplier=upper,
    )


def rank_candidate_pipeline_rows(
    rows: list[dict[str, Any]],
    *,
    confirmed_latest_families: set[str] | frozenset[str] | None = None,
    existing_by_family: Mapping[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    scored = []
    for index, row in enumerate(rows):
        acquisition = candidate_true_bug_acquisition(
            row,
            confirmed_latest_families=confirmed_latest_families or frozenset(),
            existing_by_family=existing_by_family or {},
        )
        enriched = dict(row)
        enriched["candidate_acquisition"] = acquisition
        scored.append((index, enriched))
    scored.sort(
        key=lambda item: (
            -float(item[1]["candidate_acquisition"]["acquisition_score"]),
            -float(item[1]["candidate_acquisition"]["true_bug_probability"]),
            item[0],
        )
    )
    return [row for _index, row in scored]


def candidate_true_bug_acquisition(
    row: Mapping[str, Any],
    *,
    confirmed_latest_families: set[str] | frozenset[str],
    existing_by_family: Mapping[str, list[str]],
) -> dict[str, Any]:
    findings = [finding for finding in row.get("findings", []) or [] if isinstance(finding, dict)]
    config = row.get("config", {}) if isinstance(row.get("config", {}), Mapping) else {}
    known_families = tuple(config.get("known_saturated_bug_families", []) or ())
    families = list(row.get("families", []) or candidate_issue_family_keys(findings, known_families).keys())
    rewardable_count = sum(is_rewardable_candidate_issue_finding(finding, known_families) for finding in findings)
    false_positive_count = sum(is_false_positive_finding(finding) for finding in findings)
    issue_replay_count = sum(is_issue_replay_finding(finding) for finding in findings)
    source_issue_count = sum(has_source_issue(finding) for finding in findings)
    duplicate_family_count = sum(1 for family in families if existing_by_family.get(family))
    confirmed_family_count = sum(1 for family in families if family in confirmed_latest_families)
    evidence_completeness = _evidence_completeness(row)
    recheck_prior = _recheck_prior(row.get("candidate_recheck", {}))
    novelty_probability = 1.0
    if families and (duplicate_family_count or confirmed_family_count):
        novelty_probability = max(0.05, 1.0 - ((duplicate_family_count + confirmed_family_count) / len(families)))
    reduction_potential = _reduction_potential(row.get("case", {}))
    proof_potential = (0.50 * evidence_completeness) + (0.35 * recheck_prior) + (0.15 * reduction_potential)
    penalty = (
        1.50 * false_positive_count
        + 1.00 * issue_replay_count
        + 1.00 * source_issue_count
        + 0.65 * duplicate_family_count
        + 0.85 * confirmed_family_count
    )
    utility = (
        2.75 * rewardable_count
        + 1.15 * novelty_probability
        + 1.25 * proof_potential
        + 0.45 * reduction_potential
        - penalty
    )
    probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, utility))))
    return {
        "schema_version": BUG_DISCOVERY_SYSTEM_SCHEMA_VERSION,
        "acquisition_score": max(0.0, utility),
        "true_bug_probability": probability,
        "rewardable_candidate_finding_count": rewardable_count,
        "false_positive_finding_count": false_positive_count,
        "issue_replay_finding_count": issue_replay_count,
        "source_issue_finding_count": source_issue_count,
        "duplicate_family_count": duplicate_family_count,
        "confirmed_latest_family_count": confirmed_family_count,
        "novelty_probability": novelty_probability,
        "evidence_completeness": evidence_completeness,
        "recheck_prior": recheck_prior,
        "reduction_potential": reduction_potential,
        "proof_potential": proof_potential,
        "families": families,
    }


def _counter_from_mapping(value: Any) -> Counter[str]:
    if not isinstance(value, Mapping):
        return Counter()
    return Counter(
        {
            str(key): count_int
            for key, count in value.items()
            if (count_int := _nonnegative_int(count, 0)) > 0
        }
    )


def _evidence_completeness(row: Mapping[str, Any]) -> float:
    checks = [
        bool(row.get("case")),
        bool(row.get("findings")),
        bool(row.get("normalized")),
        bool(row.get("raw_results")),
        bool(str(row.get("bug_dir", "") or "").strip()) or (bool(row.get("normalized")) and bool(row.get("raw_results"))),
    ]
    return sum(int(check) for check in checks) / float(len(checks))


def _recheck_prior(value: Any) -> float:
    if not isinstance(value, Mapping) or not value:
        return 0.35
    attempts = _nonnegative_int(value.get("attempts", 0), 0)
    if bool(value.get("reproduced", False)):
        return 1.0
    non_reproduced = value.get("non_reproduced_keys", []) or []
    reproduced = value.get("reproduced_keys", []) or []
    if attempts > 0 and not non_reproduced:
        return 0.85
    if reproduced:
        return 0.70
    if attempts > 0:
        return 0.10
    return 0.35


def _reduction_potential(case: Any) -> float:
    if not isinstance(case, Mapping):
        return 0.0
    tables = case.get("tables", []) or []
    row_count = 0
    for table in tables:
        if isinstance(table, Mapping):
            row_count += len(table.get("rows", []) or [])
    program = case.get("program", {}) if isinstance(case.get("program", {}), Mapping) else {}
    operations = program.get("operations", []) or []
    complexity = row_count + len(operations)
    if complexity <= 0:
        return 0.25
    return min(1.0, math.log1p(complexity) / math.log(32.0))


def _nonnegative_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default)
    if not math.isfinite(number):
        number = float(default)
    return max(0.0, number)


def _nonnegative_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)
    return max(0, number)


def _bounded_boltzmann_allocation(
    weights: list[float],
    *,
    min_multiplier: float,
    max_multiplier: float,
) -> list[float]:
    count = len(weights)
    if count <= 0:
        return []
    target_total = float(count)
    if min_multiplier * count > target_total:
        return [target_total / count for _weight in weights]
    if max_multiplier * count < target_total:
        return [target_total / count for _weight in weights]
    if sum(weights) <= 0.0:
        return [1.0 for _weight in weights]

    def allocated(scale: float) -> list[float]:
        return [max(min_multiplier, min(max_multiplier, scale * weight)) for weight in weights]

    low = 0.0
    high = 1.0
    while sum(allocated(high)) < target_total:
        high *= 2.0
    for _iteration in range(80):
        mid = (low + high) / 2.0
        if sum(allocated(mid)) < target_total:
            low = mid
        else:
            high = mid
    return allocated(high)
