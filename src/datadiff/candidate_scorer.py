from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from datadiff.rust_kernel import score_candidate_feature_metrics_batch


CandidateFeatureMetrics = tuple[float, float, float, float, float, int, int, float, float]


@dataclass(slots=True)
class CandidateScoringContext:
    feature_counts: Counter[str]
    finding_feature_counts: Counter[str]
    root_cause_counts: Counter[str]
    frontier_bucket_counts: Counter[str]
    discovery_bucket_counts: Counter[str]
    discovery_bucket_signal_counts: Counter[str]
    recent_discovery_stale_counts: Counter[str]
    recent_discovery_signal_counts: Counter[str]
    candidate_bug_root_hits: Counter[str]
    known_saturated_roots: set[str]
    issue_replay_root_hits: Counter[str]
    issue_inspired_source_counts: Counter[str]
    issue_replay_count: int
    enable_family_saturation: bool
    family_saturation_threshold: int
    family_saturation_penalty: float
    issue_replay_saturation_threshold: int
    issue_replay_saturation_penalty: float
    issue_replay_global_saturation_threshold: int
    issue_replay_global_saturation_penalty: float
    issue_inspired_source_saturation_threshold: int
    issue_inspired_source_saturation_penalty: float
    recent_discovery_window_count: int
    online_weight_updates: int
    online_multiplier_for_prefix: Callable[[str, str], float]
    compiled_discovery_biases: Any
    discovery_bias_evaluator: Callable[[Any, Any], tuple[float, float, float, float, bool, list[str]]]
    discovery_metrics: Callable[..., tuple[float, float, float]]
    target_template_bonus_from_features: Callable[..., float]
    target_no_yield_penalty: Callable[..., float]


@dataclass(slots=True)
class DenseCandidateScore:
    score: float
    discovery_bias_hits: list[str]
    frontier_conformance_metric: float = 0.0
    contribution_potential_metric: float = 0.0
    target_priority_metric: float = 0.0
    specific_target_matches_metric: float = 0.0
    discovery_bias_bonus_metric: float = 0.0
    discovery_diversity_bonus_metric: float = 0.0
    profile_saturation_penalty_metric: float = 0.0
    target_no_yield_penalty_metric: float = 0.0
    discovery_stale_penalty_metric: float = 0.0
    recent_discovery_loop_penalty_metric: float = 0.0
    resolved_semantic_boundary_penalty_metric: float = 0.0
    candidate_pool_bias_bonus_metric: float = 0.0
    discovery_bias_keep_in_pool_flag: bool = False
    family_saturation_active_flag: bool = False
    profile_saturation_active_flag: bool = False
    discovery_stale_active_flag: bool = False
    recent_discovery_loop_active_flag: bool = False
    issue_replay_global_saturation_active_flag: bool = False
    issue_inspired_source_saturation_active_flag: bool = False
    path_coverage_proxy_metric: float = 0.0
    data_sensitivity_metric: float = 0.0
    target_bonus_metric: float = 0.0
    target_template_matches_metric: float = 0.0
    target_template_bonus_metric: float = 0.0
    finding_yield_bonus_metric: float = 0.0
    combo_priority_metric: float = 0.0
    online_weight_mean_metric: float = 1.0
    online_weight_max_metric: float = 1.0
    feature_saturation_penalty_metric: float = 0.0
    root_saturation_penalty_metric: float = 0.0
    family_saturation_penalty_metric: float = 0.0
    issue_replay_saturation_penalty_metric: float = 0.0
    issue_replay_global_saturation_penalty_metric: float = 0.0
    issue_inspired_source_saturation_penalty_metric: float = 0.0
    issue_replay_saturation_active_flag: bool = False


@dataclass(slots=True)
class CandidateScorer:
    context: CandidateScoringContext
    discovery_bucket_metric_cache: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def score(self, analysis: Any) -> DenseCandidateScore:
        return score_candidate(
            analysis,
            self.context,
            discovery_bucket_metric_cache=self.discovery_bucket_metric_cache,
        )

    def score_many(self, analyses: list[Any] | tuple[Any, ...]) -> list[DenseCandidateScore]:
        feature_metrics = _score_candidate_feature_metrics_many(analyses, self.context)
        return [
            score_candidate(
                analysis,
                self.context,
                discovery_bucket_metric_cache=self.discovery_bucket_metric_cache,
                feature_metrics=metrics,
            )
            for analysis, metrics in zip(analyses, feature_metrics, strict=True)
        ]


def score_candidate(
    analysis: Any,
    context: CandidateScoringContext,
    *,
    discovery_bucket_metric_cache: dict[str, tuple[float, float, float]] | None = None,
    feature_metrics: CandidateFeatureMetrics | None = None,
) -> DenseCandidateScore:
    feature_score_specs = analysis.feature_score_specs
    matched_targets = analysis.matched_targets
    matched_target_count = analysis.matched_target_count
    specific_target_matches = analysis.specific_target_match_count
    target_template_matches = analysis.target_template_match_count
    target_template_bonus = context.target_template_bonus_from_features(
        analysis.target_template_features,
        context.feature_counts,
    )
    target_priority = analysis.target_match_priority_base + target_template_bonus
    target_no_yield_penalty = context.target_no_yield_penalty(
        analysis.canonical_features,
        matched_targets,
        context.feature_counts,
        context.finding_feature_counts,
        expanded_features=analysis.expanded_features,
        target_penalty_specs=analysis.target_penalty_specs,
    )
    root_cause_counts_get = context.root_cause_counts.get
    if feature_metrics is None:
        feature_metrics = _score_candidate_feature_metrics(feature_score_specs, context)
    (
        path_novelty_total,
        data_weighted_total,
        finding_yield_total,
        feature_saturation_total,
        profile_saturation_penalty,
        path_novelty_count,
        data_novelty_count,
        online_weight_total,
        online_weight_max,
    ) = feature_metrics

    path_coverage_proxy = 0.0
    if analysis.path_feature_weight_bases:
        path_coverage_proxy = (
            path_novelty_total / analysis.path_feature_count_sqrt
            + analysis.path_operation_diversity_bonus
            + analysis.path_sequence_bonus
        )
    data_sensitivity = 0.0
    if analysis.data_feature_weight_bases:
        data_sensitivity = (data_weighted_total / analysis.data_feature_count_sqrt) * 0.35

    frontier_buckets = analysis.frontier_buckets
    frontier_conformance = analysis.frontier_raw_score
    frontier_novelty_count = 0
    if frontier_buckets:
        frontier_counts_get = context.frontier_bucket_counts.get
        frontier_weighted_novelty = 0.0
        for bucket in frontier_buckets:
            bucket_count = frontier_counts_get(bucket, 0)
            frontier_weighted_novelty += 1.0 / (1.0 + bucket_count)
            if bucket_count == 0:
                frontier_novelty_count += 1
        frontier_conformance += (frontier_weighted_novelty / analysis.frontier_bucket_count_sqrt) * 0.35

    (
        discovery_bias_bonus,
        discovery_bias_novelty_bonus,
        discovery_bias_contribution_bonus,
        discovery_bias_pool_bonus,
        discovery_bias_keep_in_pool,
        discovery_bias_hits,
    ) = context.discovery_bias_evaluator(analysis, context.compiled_discovery_biases)
    (
        discovery_diversity_bonus,
        discovery_stale_penalty,
        recent_discovery_loop_penalty,
    ) = context.discovery_metrics(
        analysis.discovery_bucket_weights,
        context.discovery_bucket_counts,
        context.discovery_bucket_signal_counts,
        context.recent_discovery_stale_counts,
        context.recent_discovery_signal_counts,
        recent_window_count=context.recent_discovery_window_count,
        bucket_metric_cache=discovery_bucket_metric_cache,
    )
    discovery_diversity_bonus += discovery_bias_novelty_bonus
    discovery_stale_active = discovery_stale_penalty > 0.0
    recent_discovery_loop_active = recent_discovery_loop_penalty > 0.0

    target_bonus = 3.0 * target_priority
    finding_yield_bonus = (finding_yield_total / analysis.canonical_feature_count_sqrt) * 0.50
    feature_saturation_penalty = (feature_saturation_total / analysis.canonical_feature_count_sqrt) * 0.08
    profile_saturation_active = profile_saturation_penalty > 0.0

    predicted_roots = analysis.predicted_roots
    root_saturation_penalty = 0.0
    root_novelty = 0
    root_saturation = 0
    for root in predicted_roots:
        root_hits = root_cause_counts_get(root, 0)
        root_saturation_penalty += _root_saturation(root_hits)
        if root_hits == 0:
            root_novelty += 1
        elif root_hits >= 12:
            root_saturation += 1

    resolved_semantic_boundary_penalty = analysis.resolved_semantic_boundary_penalty
    family_saturation_penalty = 0.0
    family_saturation_active = False
    issue_replay_saturation_penalty = 0.0
    issue_replay_saturation_active = False
    issue_replay_global_saturation_penalty = 0.0
    issue_replay_global_saturation_active = False
    issue_inspired_source_saturation_penalty = 0.0
    issue_inspired_source_saturation_active = False
    if context.enable_family_saturation:
        family_saturation_penalty, family_saturation_active = _predicted_family_saturation_state(
            predicted_roots,
            root_hit_counts=context.candidate_bug_root_hits,
            known_saturated_roots=context.known_saturated_roots,
            threshold=context.family_saturation_threshold,
            penalty_weight=context.family_saturation_penalty,
        )
        issue_replay_saturation_penalty, issue_replay_saturation_active = _predicted_family_saturation_state(
            predicted_roots,
            root_hit_counts=context.issue_replay_root_hits,
            known_saturated_roots=_EMPTY_ROOTS,
            threshold=context.issue_replay_saturation_threshold,
            penalty_weight=context.issue_replay_saturation_penalty,
        )
        if analysis.has_issue_replay_source:
            issue_replay_global_saturation_penalty = _family_saturation_penalty(
                context.issue_replay_count,
                threshold=context.issue_replay_global_saturation_threshold,
                penalty_weight=context.issue_replay_global_saturation_penalty,
            )
            issue_replay_global_saturation_active = _is_globally_saturated(
                context.issue_replay_count,
                threshold=context.issue_replay_global_saturation_threshold,
            )
        if analysis.has_issue_inspired_source:
            source_issue_hits = _max_source_issue_hits_for_keys(
                analysis.issue_inspired_source_keys,
                source_counts=context.issue_inspired_source_counts,
            )
            issue_inspired_source_saturation_penalty = _family_saturation_penalty(
                source_issue_hits,
                threshold=context.issue_inspired_source_saturation_threshold,
                penalty_weight=context.issue_inspired_source_saturation_penalty,
            )
            issue_inspired_source_saturation_active = _is_globally_saturated(
                source_issue_hits,
                threshold=context.issue_inspired_source_saturation_threshold,
            )

    if matched_targets:
        saturation_multiplier = 0.35 if specific_target_matches else 0.85
        feature_saturation_penalty *= saturation_multiplier
        root_saturation_penalty *= saturation_multiplier
        profile_saturation_penalty *= saturation_multiplier
        family_saturation_penalty *= saturation_multiplier
        issue_replay_saturation_penalty *= saturation_multiplier
        issue_replay_global_saturation_penalty *= saturation_multiplier
        issue_inspired_source_saturation_penalty *= saturation_multiplier

    issue_replay_saturation_active_any = (
        issue_replay_saturation_active or issue_replay_global_saturation_active
    )
    contribution_potential = (
        frontier_conformance
        + 0.30 * path_novelty_count
        + 0.20 * data_novelty_count
        + 0.45 * frontier_novelty_count
        + 0.35 * root_novelty
        + 0.60 * matched_target_count
        - 0.20 * root_saturation
        + discovery_bias_contribution_bonus
    )
    combo_priority = analysis.combo_priority_base
    online_weight_mean = (
        online_weight_total / analysis.learnable_feature_count
        if analysis.learnable_feature_count
        else 1.0
    )
    score = (
        path_coverage_proxy
        + data_sensitivity
        + frontier_conformance
        + discovery_diversity_bonus
        + target_bonus
        + finding_yield_bonus
        + combo_priority
        + discovery_bias_bonus
    )
    score -= (
        feature_saturation_penalty
        + root_saturation_penalty
        + resolved_semantic_boundary_penalty
        + profile_saturation_penalty
        + family_saturation_penalty
        + issue_replay_saturation_penalty
        + issue_replay_global_saturation_penalty
        + issue_inspired_source_saturation_penalty
        + target_no_yield_penalty
        + discovery_stale_penalty
        + recent_discovery_loop_penalty
    )

    return DenseCandidateScore(
        score=score,
        discovery_bias_hits=discovery_bias_hits,
        frontier_conformance_metric=frontier_conformance,
        contribution_potential_metric=contribution_potential,
        target_priority_metric=target_priority,
        specific_target_matches_metric=float(specific_target_matches),
        discovery_bias_bonus_metric=discovery_bias_bonus,
        discovery_diversity_bonus_metric=discovery_diversity_bonus,
        profile_saturation_penalty_metric=-profile_saturation_penalty,
        target_no_yield_penalty_metric=-target_no_yield_penalty,
        discovery_stale_penalty_metric=-discovery_stale_penalty,
        recent_discovery_loop_penalty_metric=-recent_discovery_loop_penalty,
        resolved_semantic_boundary_penalty_metric=-resolved_semantic_boundary_penalty,
        candidate_pool_bias_bonus_metric=discovery_bias_pool_bonus,
        discovery_bias_keep_in_pool_flag=discovery_bias_keep_in_pool,
        family_saturation_active_flag=(family_saturation_active or issue_replay_saturation_active_any),
        profile_saturation_active_flag=profile_saturation_active,
        discovery_stale_active_flag=discovery_stale_active,
        recent_discovery_loop_active_flag=recent_discovery_loop_active,
        issue_replay_global_saturation_active_flag=issue_replay_global_saturation_active,
        issue_inspired_source_saturation_active_flag=issue_inspired_source_saturation_active,
        path_coverage_proxy_metric=path_coverage_proxy,
        data_sensitivity_metric=data_sensitivity,
        target_bonus_metric=target_bonus,
        target_template_matches_metric=float(target_template_matches),
        target_template_bonus_metric=target_template_bonus,
        finding_yield_bonus_metric=finding_yield_bonus,
        combo_priority_metric=combo_priority,
        online_weight_mean_metric=online_weight_mean,
        online_weight_max_metric=online_weight_max,
        feature_saturation_penalty_metric=-feature_saturation_penalty,
        root_saturation_penalty_metric=-root_saturation_penalty,
        family_saturation_penalty_metric=-family_saturation_penalty,
        issue_replay_saturation_penalty_metric=-issue_replay_saturation_penalty,
        issue_replay_global_saturation_penalty_metric=-issue_replay_global_saturation_penalty,
        issue_inspired_source_saturation_penalty_metric=-issue_inspired_source_saturation_penalty,
        issue_replay_saturation_active_flag=issue_replay_saturation_active_any,
    )


def _score_candidate_feature_metrics_many(
    analyses: list[Any] | tuple[Any, ...],
    context: CandidateScoringContext,
) -> list[CandidateFeatureMetrics]:
    if not analyses:
        return []
    candidate_specs = [
        _candidate_feature_metric_specs(analysis.feature_score_specs, context)
        for analysis in analyses
    ]
    return score_candidate_feature_metrics_batch(
        candidate_specs,
        context.feature_counts,
        context.finding_feature_counts,
    )


def _candidate_feature_metric_specs(
    feature_score_specs: Any,
    context: CandidateScoringContext,
) -> tuple[tuple[str, float, float, float, float, float | None, bool], ...]:
    feature_multiplier = context.online_multiplier_for_prefix
    return tuple(
        (
            spec.feature,
            float(spec.finding_weight_base),
            float(spec.saturation_weight_base),
            float(spec.path_weight_base),
            float(spec.data_weight_base),
            (
                float(feature_multiplier(spec.feature, spec.learnable_feature_prefix))
                if spec.learnable_feature_prefix is not None
                else None
            ),
            bool(spec.is_mixed_profile),
        )
        for spec in feature_score_specs
    )


def _score_candidate_feature_metrics(
    feature_score_specs: Any,
    context: CandidateScoringContext,
) -> CandidateFeatureMetrics:
    feature_counts_get = context.feature_counts.get
    finding_feature_counts_get = context.finding_feature_counts.get
    online_weight_total = 0.0
    online_weight_max = 1.0
    path_novelty_total = 0.0
    data_weighted_total = 0.0
    finding_yield_total = 0.0
    feature_saturation_total = 0.0
    profile_saturation_penalty = 0.0
    path_novelty_count = 0
    data_novelty_count = 0
    if feature_score_specs:
        feature_multiplier = context.online_multiplier_for_prefix
        for spec in feature_score_specs:
            count = feature_counts_get(spec.feature, 0)
            if spec.learnable_feature_prefix is not None:
                multiplier_value = feature_multiplier(spec.feature, spec.learnable_feature_prefix)
                online_weight_total += multiplier_value
                if multiplier_value > online_weight_max:
                    online_weight_max = multiplier_value
            else:
                multiplier_value = 1.0
            if spec.path_weight_base > 0.0:
                path_novelty_total += (spec.path_weight_base * multiplier_value) / (1.0 + count)
                if count == 0:
                    path_novelty_count += 1
            if spec.data_weight_base > 0.0:
                data_weighted_total += (spec.data_weight_base * multiplier_value) * (
                    1.0 + 1.0 / (1.0 + count)
                )
                if count == 0:
                    data_novelty_count += 1
            finding_hits = finding_feature_counts_get(spec.feature, 0)
            finding_yield_total += (
                _bounded_finding_signal(finding_hits) * spec.finding_weight_base * multiplier_value
            )
            feature_saturation_total += (
                _feature_saturation(finding_hits) * spec.saturation_weight_base * multiplier_value
            )
            if spec.is_mixed_profile:
                profile_saturation_penalty += _profile_saturation(count)
    return (
        path_novelty_total,
        data_weighted_total,
        finding_yield_total,
        feature_saturation_total,
        profile_saturation_penalty,
        path_novelty_count,
        data_novelty_count,
        online_weight_total,
        online_weight_max,
    )


@lru_cache(maxsize=None)
def _bounded_finding_signal(count: int) -> float:
    if count <= 0:
        return 0.0
    return min(math.log1p(count), 2.0) / (1.0 + count / 50.0)


@lru_cache(maxsize=None)
def _feature_saturation(count: int) -> float:
    if count <= 25:
        return 0.0
    return math.log1p(count - 25)


@lru_cache(maxsize=None)
def _root_saturation(count: int) -> float:
    if count <= 6:
        return 0.0
    return math.log1p(count - 6) * 0.45


def _profile_saturation(count: int) -> float:
    if count <= 3:
        return 0.0
    return min(4.0, math.log1p(count - 3) * 1.15)


_EMPTY_ROOTS: frozenset[str] = frozenset()


def _predicted_family_saturation_state(
    predicted_roots: set[str],
    *,
    root_hit_counts: Counter[str],
    known_saturated_roots: set[str] | frozenset[str],
    threshold: int,
    penalty_weight: float,
) -> tuple[float, bool]:
    if threshold <= 0:
        return 0.0, False
    penalty = 0.0
    active = False
    positive_penalty_weight = max(0.0, penalty_weight)
    for root in predicted_roots:
        hit_count = _predicted_family_hit_count(
            root,
            root_hit_counts=root_hit_counts,
            known_saturated_roots=known_saturated_roots,
            threshold=threshold,
        )
        if hit_count >= threshold:
            active = True
            if positive_penalty_weight > 0.0:
                penalty += _family_saturation_penalty(
                    hit_count,
                    threshold=threshold,
                    penalty_weight=positive_penalty_weight,
                )
    return penalty, active


def _predicted_family_hit_count(
    root: str,
    *,
    root_hit_counts: Counter[str],
    known_saturated_roots: set[str] | frozenset[str],
    threshold: int,
    include_known_families: bool = True,
) -> int:
    dynamic_hits = root_hit_counts[root]
    known_hits = threshold if include_known_families and root in known_saturated_roots else 0
    return max(dynamic_hits, known_hits)


def _max_source_issue_hits_for_keys(
    source_keys: tuple[str, ...] | list[str],
    *,
    source_counts: Counter[str],
) -> int:
    return max((source_counts[source_key] for source_key in source_keys), default=0)


def _family_saturation_penalty(count: int, *, threshold: int, penalty_weight: float) -> float:
    if count < threshold:
        return 0.0
    return max(0.0, penalty_weight) * (1.0 + math.log1p(count - threshold))


def _is_globally_saturated(count: int, *, threshold: int) -> bool:
    return threshold > 0 and count >= threshold
