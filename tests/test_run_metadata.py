from types import SimpleNamespace

from datadiff.config import ExperimentConfig
from datadiff.datagen import generate_case
from datadiff.normalizer import NormalizedResult
from datadiff import runner as runner_module
from datadiff.run_metadata import (
    _attach_case_fingerprint_to_row_case,
    _attach_disagreement_descriptor_to_row_case,
    _candidate_quality_context,
    _candidate_target_keys_from_metadata,
    _feedback_target_keys,
    _fingerprint_anchor_result,
    _generated_candidate_metadata,
    _selected_candidate_metadata,
)


def test_generated_and_selected_candidate_metadata_have_stable_shape():
    case = generate_case(17, profile="common")
    config = ExperimentConfig(enable_replay_bug=False)

    generated = _generated_candidate_metadata(case)
    selected = _selected_candidate_metadata(case, config, guidance_enabled=True)

    assert generated["seed_lineage"]["root_seed"] == 17
    assert generated["seed_lineage"]["depth"] == 0
    assert generated["mutation"]["operator"] == "generated"
    assert selected["source"] == "generated"
    assert selected["generated_seed"] == 17
    assert selected["replay_filter"]["enabled"] is True
    assert selected["family_saturation_filter"]["enabled"] is True
    assert selected["operation_combo"]["template"]


def test_candidate_quality_context_dedupes_metadata_targets_and_capabilities():
    calls: list[tuple[str, ...] | None] = []

    class FakeFeedback:
        def candidate_quality_context(self, case, *, target_keys=None):
            calls.append(tuple(target_keys) if target_keys is not None else None)
            return {"archive_candidate": True, "case_id": case.case_id}

    case = generate_case(18, profile="common")
    metadata = {
        "feedback_decision": {"target_keys": ["target:a", "target:a", "", "target:b"]},
    }

    keys = _candidate_target_keys_from_metadata(metadata)
    context = _candidate_quality_context(
        FakeFeedback(),
        case,
        metadata,
        target_capabilities=["op:join", "op:join", "table:multi"],
    )

    assert keys == ["target:a", "target:b"]
    assert context == {"archive_candidate": True, "case_id": case.case_id}
    assert calls == [("target:a", "target:b", "capability:op:join", "capability:table:multi")]


def test_attach_metadata_helpers_copy_existing_metadata_before_update():
    row = {"case": {"case_id": "case-1", "metadata": {"existing": True}}}
    original_metadata = row["case"]["metadata"]

    _attach_disagreement_descriptor_to_row_case(row, {"mismatch_class": "row_count"})
    _attach_case_fingerprint_to_row_case(row, {"hash": "abc"})

    assert row["case"]["metadata"] is not original_metadata
    assert row["case"]["metadata"]["existing"] is True
    assert row["case"]["metadata"]["disagreement_descriptor"] == {"mismatch_class": "row_count"}
    assert row["case"]["metadata"]["case_fingerprint"] == {"hash": "abc"}


def test_fingerprint_anchor_result_prefers_first_sorted_ok_backend():
    normalized = {
        "z_backend": NormalizedResult("z_backend", "ok", ["x"], [{"x": 2}]),
        "a_backend": NormalizedResult("a_backend", "ok", ["x"], [{"x": 1}]),
    }
    error_only = {
        "z_backend": NormalizedResult("z_backend", "error", [], [], error="boom"),
        "a_backend": NormalizedResult("a_backend", "error", [], [], error="boom"),
    }

    assert _fingerprint_anchor_result(normalized).backend == "a_backend"
    assert _fingerprint_anchor_result(error_only).backend == "a_backend"
    assert _fingerprint_anchor_result({}) is None


def test_feedback_target_keys_include_guidance_targets_and_operation_risks():
    guidance_row = {
        "matched_targets": ["semi_anti_join_rewrite", "strings", "groupby"],
        "features": [
            "pattern:semi_anti_join_rewrite",
            "semantic_family:join_membership",
            "exploration_objective:cross_model_consistency",
            "combo_risk:groupby_aggregation",
            "rows:many",
        ],
        "config": {
            "semantic_focus_families": ["conditional_semantics"],
            "semantic_focus_signals": ["left_join_case_when_membership"],
        },
    }
    operation_combo = {
        "template": "join_filter_groupby",
        "correctness_risks": ["semi_anti_join_filter_pushdown", "groupby_aggregation"],
    }

    keys = _feedback_target_keys(
        guidance_row,
        operation_combo,
        target_capabilities=["op:join", "table:multi"],
    )

    assert "target:semi_anti_join_rewrite" in keys
    assert "feature:pattern:semi_anti_join_rewrite" in keys
    assert "feature:semantic_family:join_membership" in keys
    assert "feature:exploration_objective:cross_model_consistency" in keys
    assert "semantic_family:join_membership" in keys
    assert "exploration_objective:cross_model_consistency" in keys
    assert "combo:join_filter_groupby" in keys
    assert "semantic_signal:semi_anti_join_filter_pushdown" in keys
    assert "risk:semi_anti_join_filter_pushdown" in keys
    assert "semantic_family:conditional_semantics" in keys
    assert "semantic_signal:left_join_case_when_membership" in keys
    assert "risk:left_join_case_when_membership" in keys
    assert "capability:op:join" in keys
    assert "capability:table:multi" in keys
    assert "rows:many" not in keys
    assert "target:strings" not in keys
    assert "target:groupby" not in keys
    assert "feature:combo_risk:groupby_aggregation" not in keys


def test_runner_reexports_run_metadata_helpers_for_compatibility():
    assert runner_module._feedback_target_keys is _feedback_target_keys
    assert runner_module._generated_candidate_metadata is _generated_candidate_metadata
    assert runner_module._candidate_quality_context is _candidate_quality_context
