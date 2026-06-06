from types import SimpleNamespace

from datadiff import runner as runner_module
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.run_feedback import (
    _feedback_storage_decision,
    _source_scheduler_snapshot,
    apply_feedback_updates,
)


def _case(seed: int = 1, operation: dict | None = None) -> Case:
    return Case(
        f"case-{seed}",
        seed,
        [TableData("t0", [ColumnSpec("x", "int")], [{"x": 1}, {"x": None}])],
        Program(
            f"prog-{seed}",
            seed,
            [operation or {"op": "filter", "column": "x", "cmp": ">", "value": 0}],
        ),
    )


def _base_row() -> dict:
    return {
        "candidate_source": "generated",
        "is_new_behavior": True,
        "signal_new_behavior": True,
        "duration_ms": 125.0,
        "preflight": {"valid": True, "fallback_used": False},
        "findings": [],
        "quality_oracles": [],
        "generator_profile_selection": {
            "strategy": "contextual_bandit",
            "profile": "common",
        },
        "semantic_objective_selection": {
            "strategy": "contextual_bandit",
            "scope": "semantic_objective",
            "action": "objective:null_boundary",
        },
        "metamorphic_relation_selection": {
            "strategy": "contextual_bandit",
            "scope": "metamorphic_relation",
            "action": "row_order_invariance",
        },
        "version_pair_selection": {
            "strategy": "contextual_bandit",
            "scope": "version_pair",
            "action": "latest->fixed",
        },
        "backend_pair_selection": {
            "strategy": "contextual_bandit",
            "scope": "backend_pair",
            "action_pool": ["duckdb|pandas"],
            "priority": ["duckdb|pandas"],
        },
        "backend_pair_context": ["root:filter"],
        "disagreement_descriptor": {
            "pair_disagrees": [{"left": "duckdb", "right": "pandas", "disagrees": True}],
        },
        "case_learning_context": ["op:filter"],
    }


def test_feedback_storage_decision_skips_feedback_children_and_probe_cases():
    probe_case = _case(2, {"op": "running_sum", "source": "x", "as": "run_x"})
    ordinary_case = _case(3)

    assert _feedback_storage_decision(probe_case) == (False, "calibration_probe_case")
    assert _feedback_storage_decision(
        ordinary_case,
        candidate_source="feedback_mutation",
        seed_lineage={"depth": 0},
    ) == (False, "feedback_mutation_child")
    assert _feedback_storage_decision(
        ordinary_case,
        candidate_source="generated",
        seed_lineage={"depth": 1},
    ) == (False, "feedback_mutation_child")
    assert _feedback_storage_decision(ordinary_case) == (True, "")


def test_apply_feedback_updates_disabled_path_populates_stable_row_fields():
    row = _base_row()

    result = apply_feedback_updates(
        row=row,
        case=_case(),
        feedback=None,
        config=ExperimentConfig(enable_feedback=False),
        selected_meta={"source": "generated", "seed_lineage": {}, "operation_combo": {}},
        guidance_row={},
        preflight_row={"valid": True},
        behavior_signature="behavior",
        signal_signature="signal",
        discovery_signature="discovery",
        version_id="latest->fixed",
        generator_profile_context_features=(),
        target_capabilities=(),
    )

    assert result.finding_outcomes is None
    assert result.elapsed_ms == 0.0
    assert row["stored_in_feedback_corpus"] is False
    assert row["feedback_corpus_persisted"] is False
    assert row["feedback_eligible"] is False
    assert row["feedback_skip_reason"] == "feedback_disabled"
    assert row["feedback_record_skip_reason"] == ""
    assert row["source_reward"] is None
    assert row["backend_pair_feedback"] == {"recorded": 0, "rewarded_pairs": [], "disagree_pairs": []}
    assert row["source_scheduler"] == []
    assert row["feedback_summary"]["candidate_source"] == "generated"


def test_apply_feedback_updates_records_corpus_and_learning_feedback():
    class FakeScheduler:
        def __init__(self):
            self.calls = []

        def record_result(self, source, **kwargs):
            self.calls.append((source, kwargs))
            return 1.25

        def snapshot(self):
            return [{"source": "generated", "score": 1.25}]

    class FakeLearning:
        def __init__(self):
            self.calls = []

        def record_outcome(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    class FakeFeedback:
        def __init__(self):
            self.source_scheduler = FakeScheduler()
            self.adaptive_learning = FakeLearning()
            self.last_persisted_to_disk = False
            self.last_record_skip_reason = ""
            self.record_calls = []

        def record(self, case, behavior_signature, has_finding, **kwargs):
            self.record_calls.append((case, behavior_signature, has_finding, kwargs))
            self.last_persisted_to_disk = True
            self.last_record_skip_reason = ""
            return True

        def record_candidate_outcome(self, candidate_source, **kwargs):
            return self.source_scheduler.record_result(candidate_source, **kwargs)

    row = _base_row()
    row["findings"] = [
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "filter_null_semantics",
            "suspicious_backends": ["duckdb"],
            "signature": "sig-1",
        }
    ]
    row["case_fingerprint"] = {"hash": "abc"}
    row["disagreement_descriptor"]["mismatch_class"] = "row_count"
    feedback = FakeFeedback()

    result = apply_feedback_updates(
        row=row,
        case=_case(),
        feedback=feedback,
        config=ExperimentConfig(enable_feedback=True),
        selected_meta={
            "source": "generated",
            "seed_lineage": {},
            "operation_combo": {
                "template": "filter_null",
                "correctness_risks": ["null_filter_semantics"],
            },
            "generator_profile_selection": row["generator_profile_selection"],
        },
        guidance_row={
            "matched_targets": ["filter", "nulls"],
            "features": ["semantic_family:null_boundary"],
        },
        preflight_row={"valid": True, "fallback_used": False},
        behavior_signature="behavior",
        signal_signature="signal",
        discovery_signature="discovery",
        version_id="latest->fixed",
        generator_profile_context_features=("profile:common",),
        target_capabilities=("op:filter",),
    )

    assert result.finding_outcomes is not None
    assert result.elapsed_ms >= 0.0
    assert row["feedback_eligible"] is True
    assert row["feedback_skip_reason"] == ""
    assert row["stored_in_feedback_corpus"] is True
    assert row["feedback_corpus_persisted"] is True
    assert row["feedback_record_skip_reason"] == ""
    assert row["source_reward"] == 1.25
    assert row["source_scheduler"] == [{"source": "generated", "score": 1.25}]
    assert row["feedback_summary"]["candidate_bug"] is True
    assert row["feedback_summary"]["stored_in_feedback_corpus"] is True
    assert row["generator_profile_selection"]["reward"] > 0.0
    assert row["semantic_objective_selection"]["reward"] > 0.0
    assert row["metamorphic_relation_selection"]["reward"] > 0.0
    assert row["version_pair_selection"]["reward"] > 0.0
    assert row["backend_pair_feedback"]["recorded"] == 1
    assert row["backend_pair_feedback"]["disagree_pairs"] == ["duckdb|pandas"]

    case, behavior_signature, has_finding, record_kwargs = feedback.record_calls[0]
    assert case.case_id == "case-1"
    assert behavior_signature == "behavior"
    assert has_finding is True
    assert record_kwargs["novelty_signature"] == "signal"
    assert record_kwargs["discovery_signature"] == "discovery"
    assert record_kwargs["candidate_bug_families"] == ["filter_null_semantics@duckdb"]
    assert "feature:semantic_family:null_boundary" in record_kwargs["target_keys"]
    assert "combo:filter_null" in record_kwargs["target_keys"]
    assert "risk:null_filter_semantics" in record_kwargs["target_keys"]
    assert "semantic_signal:null_filter_semantics" in record_kwargs["target_keys"]
    assert "capability:op:filter" in record_kwargs["target_keys"]
    assert record_kwargs["case_fingerprint"] == {"hash": "abc"}

    source_call = feedback.source_scheduler.calls[0]
    assert source_call[0] == "generated"
    assert source_call[1]["candidate_bug"] is True
    assert source_call[1]["candidate_bug_families"] == ["filter_null_semantics@duckdb"]
    assert source_call[1]["candidate_bug_signatures"] == ["sig-1"]

    learned = [(args[0], args[1]) for args, _kwargs in feedback.adaptive_learning.calls]
    assert ("generator_profile", "common") in learned
    assert ("semantic_objective", "objective:null_boundary") in learned
    assert ("metamorphic_relation", "row_order_invariance") in learned
    assert ("version_pair", "latest->fixed") in learned
    assert ("backend_pair", "duckdb|pandas") in learned


def test_apply_feedback_updates_keeps_plain_feedback_children_out_of_corpus():
    class FakeFeedback:
        def __init__(self):
            self.last_persisted_to_disk = False
            self.last_record_skip_reason = ""
            self.record_calls = []
            self.outcome_calls = []

        def record(self, *args, **kwargs):
            self.record_calls.append((args, kwargs))
            return True

        def record_candidate_outcome(self, candidate_source, **kwargs):
            self.outcome_calls.append((candidate_source, kwargs))
            return -0.1

    row = _base_row()
    feedback = FakeFeedback()

    apply_feedback_updates(
        row=row,
        case=_case(),
        feedback=feedback,
        config=ExperimentConfig(enable_feedback=True),
        selected_meta={
            "source": "feedback_mutation",
            "seed_lineage": {"depth": 1, "parent_case_id": "case-parent"},
            "operation_combo": {},
            "generator_profile_selection": row["generator_profile_selection"],
        },
        guidance_row={},
        preflight_row={"valid": True, "fallback_used": False},
        behavior_signature="behavior",
        signal_signature="signal",
        discovery_signature="discovery",
        version_id="latest->fixed",
        generator_profile_context_features=(),
        target_capabilities=(),
    )

    assert row["feedback_eligible"] is False
    assert row["feedback_skip_reason"] == "feedback_mutation_child"
    assert row["stored_in_feedback_corpus"] is False
    assert feedback.record_calls == []
    assert feedback.outcome_calls[0][0] == "feedback_mutation"


def test_apply_feedback_updates_keeps_candidate_bug_feedback_child_in_corpus():
    class FakeFeedback:
        def __init__(self):
            self.last_persisted_to_disk = False
            self.last_record_skip_reason = ""
            self.record_calls = []
            self.outcome_calls = []

        def record(self, case, behavior_signature, has_finding, **kwargs):
            self.record_calls.append((case, behavior_signature, has_finding, kwargs))
            self.last_persisted_to_disk = True
            return True

        def record_candidate_outcome(self, candidate_source, **kwargs):
            self.outcome_calls.append((candidate_source, kwargs))
            return 4.0

    row = _base_row()
    row["findings"] = [
        {
            "triage_verdict": "candidate_implementation_bug",
            "root_cause": "groupby_aggregation",
            "suspicious_backends": ["datafusion"],
            "signature": "sig-new-child",
            "discovery_origin": "organic",
        }
    ]
    feedback = FakeFeedback()

    apply_feedback_updates(
        row=row,
        case=_case(),
        feedback=feedback,
        config=ExperimentConfig(enable_feedback=True),
        selected_meta={
            "source": "feedback_mutation",
            "seed_lineage": {"depth": 1, "parent_case_id": "case-parent"},
            "operation_combo": {},
            "generator_profile_selection": row["generator_profile_selection"],
        },
        guidance_row={},
        preflight_row={"valid": True, "fallback_used": False},
        behavior_signature="behavior",
        signal_signature="signal",
        discovery_signature="discovery",
        version_id="latest->fixed",
        generator_profile_context_features=(),
        target_capabilities=(),
    )

    assert row["feedback_eligible"] is True
    assert row["feedback_skip_reason"] == ""
    assert row["stored_in_feedback_corpus"] is True
    assert row["feedback_corpus_persisted"] is True
    case, behavior_signature, has_finding, record_kwargs = feedback.record_calls[0]
    assert case.case_id == "case-1"
    assert behavior_signature == "behavior"
    assert has_finding is True
    assert record_kwargs["candidate_bug_families"] == ["groupby_aggregation@datafusion"]
    assert feedback.outcome_calls[0][1]["candidate_bug"] is True


def test_runner_reexports_feedback_helpers_for_compatibility():
    assert runner_module._feedback_storage_decision is _feedback_storage_decision
    assert runner_module._source_scheduler_snapshot is _source_scheduler_snapshot
    assert _source_scheduler_snapshot(SimpleNamespace(source_scheduler=None)) == []
