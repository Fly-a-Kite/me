from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.quality_oracles import feedback_oracle, guidance_oracle, mutation_oracle


def _case() -> Case:
    return Case(
        "case-x",
        1,
        [
            TableData(
                "t0",
                [ColumnSpec("x", "int"), ColumnSpec("g", "str")],
                [{"x": 1, "g": "a"}, {"x": 2, "g": "b"}],
            )
        ],
        Program(
            "prog-x",
            1,
            [{"op": "groupby", "keys": ["g"], "aggs": [{"column": "x", "func": "sum", "as": "sum_x"}]}],
        ),
    )


def test_mutation_oracle_marks_productive_feedback_mutation():
    result = mutation_oracle(
        _case(),
        {"is_new_behavior": True, "findings": []},
        candidate_source="feedback_mutation",
        preflight={"valid": True, "repaired": False, "fallback_used": False},
    )

    assert result.verdict == "productive_mutation"
    assert result.passed is True


def test_feedback_oracle_marks_redundant_behavior():
    result = feedback_oracle(
        {
            "is_new_behavior": False,
            "findings": [],
            "stored_in_feedback_corpus": False,
            "behavior_signature": "abc",
        }
    )

    assert result.verdict == "redundant_behavior"
    assert result.passed is False


def test_guidance_oracle_checks_target_hit_and_productivity():
    result = guidance_oracle(
        _case(),
        {"is_new_behavior": True, "findings": []},
        guidance_decision={
            "score": 3.0,
            "matched_targets": ["groupby"],
            "candidate_count": 4,
            "contributing_candidate_count": 2,
            "pruned_candidate_count": 2,
            "frontier_buckets": ["groupby:mixed-cardinality"],
            "score_breakdown": {"frontier_conformance": 0.9, "contribution_potential": 1.4},
        },
        guidance_strategy="guided",
        guidance_targets=["groupby"],
    )

    assert result.verdict == "guided_productive"
    assert result.passed is True
    assert result.metrics["matched_targets"] == ["groupby"]
    assert result.metrics["pruned_candidate_count"] == 2
    assert result.metrics["frontier_conformance"] == 0.9


def test_guidance_oracle_respects_canonical_semantic_focus_targets():
    result = guidance_oracle(
        _case(),
        {"is_new_behavior": True, "findings": []},
        guidance_decision={
            "score": 2.0,
            "matched_targets": ["semantic_family:aggregation_cardinality"],
        },
        guidance_strategy="guided",
        guidance_targets=["semantic_family:aggregation_cardinality"],
    )

    assert result.verdict == "guided_productive"
    assert result.passed is True
    assert result.metrics["configured_targets"] == ["semantic_family:aggregation_cardinality"]


def test_quality_oracles_prefer_signal_new_behavior_over_raw_new_behavior():
    mutation = mutation_oracle(
        _case(),
        {"is_new_behavior": True, "signal_new_behavior": False, "findings": []},
        candidate_source="feedback_mutation",
        preflight={"valid": True, "repaired": False, "fallback_used": False},
    )
    feedback = feedback_oracle(
        {
            "is_new_behavior": True,
            "signal_new_behavior": False,
            "findings": [],
            "stored_in_feedback_corpus": False,
            "behavior_signature": "abc",
        }
    )
    guidance = guidance_oracle(
        _case(),
        {"is_new_behavior": True, "signal_new_behavior": False, "findings": []},
        guidance_decision={"score": 3.0, "matched_targets": ["groupby"]},
        guidance_strategy="guided",
        guidance_targets=["groupby"],
    )

    assert mutation.verdict == "redundant_mutation"
    assert feedback.verdict == "redundant_behavior"
    assert guidance.verdict == "guided_redundant"
