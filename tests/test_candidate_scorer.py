from datadiff.candidate_scorer import CandidateScorer
import datadiff.candidate_scorer as candidate_scorer_module
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.guidance import GuidanceDecision, GuidanceState
import pytest


def _case(seed: int, operations: list[dict]) -> Case:
    table = TableData(
        name="t0",
        columns=[
            ColumnSpec("id", "int", nullable=False),
            ColumnSpec("g", "str", nullable=True),
            ColumnSpec("x", "int", nullable=True),
        ],
        rows=[
            {"id": 0, "g": "alpha", "x": 1},
            {"id": 1, "g": None, "x": None},
            {"id": 2, "g": "beta", "x": -1},
        ],
    )
    return Case(
        case_id=f"case-{seed:08d}",
        seed=seed,
        tables=[table],
        program=Program(program_id=f"prog-{seed:08d}", seed=seed, operations=operations),
    )


def test_candidate_scorer_matches_guidance_decision_dense_metrics():
    case = _case(
        1,
        [
            {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
            {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            {"op": "sort", "columns": ["m_0", "id"], "ascending": True},
            {"op": "limit", "n": 2},
        ],
    )
    guidance = GuidanceState(targets=["topk", "common_workflow"])
    guidance.record_result(
        case,
        {
            "findings": [{"kind": "semantic_output_mismatch", "root_cause": "topk_ordering"}],
            "is_new_behavior": True,
            "preflight": {"valid": True, "fallback_used": False},
        },
    )
    analysis = guidance._case_analysis(case)
    scorer = CandidateScorer(guidance._candidate_scoring_context(recent_discovery_window_count=0))

    dense = scorer.score(analysis)
    decision = guidance._score_case(case, 1, recent_discovery_window_count=0)

    assert isinstance(decision, GuidanceDecision)
    assert dense.score == decision.score
    assert dense.discovery_bias_hits == decision.discovery_bias_hits
    assert dense.path_coverage_proxy_metric == decision.score_breakdown["path_coverage_proxy"]
    assert dense.data_sensitivity_metric == decision.score_breakdown["data_sensitivity"]
    assert dense.frontier_conformance_metric == decision.score_breakdown["frontier_conformance"]
    assert dense.online_weight_mean_metric == decision.score_breakdown["online_weight_mean"]
    assert dense.target_bonus_metric == decision.score_breakdown["target_bonus"]


def test_candidate_scorer_keeps_report_materialization_out_of_hot_path():
    case = _case(2, [{"op": "select", "columns": ["id"]}])
    guidance = GuidanceState()
    analysis = guidance._case_analysis(case)

    dense = CandidateScorer(guidance._candidate_scoring_context()).score(analysis)

    assert not hasattr(dense, "score_breakdown")
    assert not hasattr(dense, "online_weights")


def test_candidate_scorer_scores_candidate_pool_with_shared_context():
    cases = [
        _case(3, [{"op": "select", "columns": ["id"]}]),
        _case(
            4,
            [
                {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
                {"op": "sort", "columns": ["x", "id"], "ascending": True},
                {"op": "limit", "n": 1},
            ],
        ),
    ]
    guidance = GuidanceState(targets=["topk"])
    analyses = [guidance._case_analysis(case) for case in cases]
    scorer = CandidateScorer(guidance._candidate_scoring_context(recent_discovery_window_count=0))

    batch_scores = scorer.score_many(analyses)
    single_scores = [
        CandidateScorer(guidance._candidate_scoring_context(recent_discovery_window_count=0)).score(analysis)
        for analysis in analyses
    ]
    decision = guidance.choose_case(cases)

    assert [score.score for score in batch_scores] == pytest.approx([score.score for score in single_scores])
    assert decision.case in cases
    assert decision.candidate_count == 2
    assert decision.score_breakdown


def test_candidate_scorer_score_many_uses_batch_feature_metric_helper(monkeypatch):
    cases = [
        _case(5, [{"op": "select", "columns": ["id"]}]),
        _case(
            6,
            [
                {"op": "filter", "column": "x", "cmp": ">=", "value": 0},
                {"op": "mutate", "column": "m_0", "expr": {"kind": "add_const", "source": "x", "value": 1}},
            ],
        ),
    ]
    guidance = GuidanceState(targets=["common_workflow"])
    analyses = [guidance._case_analysis(case) for case in cases]
    original = candidate_scorer_module.score_candidate_feature_metrics_batch
    calls: list[int] = []

    def spy(candidate_specs, feature_counts, finding_feature_counts):
        calls.append(len(candidate_specs))
        return original(candidate_specs, feature_counts, finding_feature_counts)

    monkeypatch.setattr(candidate_scorer_module, "score_candidate_feature_metrics_batch", spy)

    scores = CandidateScorer(guidance._candidate_scoring_context()).score_many(analyses)

    assert len(scores) == 2
    assert calls == [2]
