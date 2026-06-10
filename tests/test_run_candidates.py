from types import SimpleNamespace

from datadiff import runner as runner_module
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case, ColumnSpec, Program, TableData
from datadiff.run_candidates import (
    _generate_case_with_optional_schema,
    _known_replay_source_filter_reason,
    _replay_bug_filter_reason,
    generate_candidate_batch,
)


def _case(seed: int, *, profile: str = "common") -> Case:
    return Case(
        case_id=f"case-{seed}-{profile}",
        seed=seed,
        tables=[TableData("t0", [ColumnSpec("x", "int")], [{"x": seed}])],
        program=Program(f"prog-{seed}", seed, [{"op": "select", "columns": ["x"]}]),
        metadata={"generator_profile": profile},
    )


def test_generate_candidate_batch_builds_pool_and_metadata_without_filters():
    calls: list[dict] = []

    def fake_generate_case(seed, *, type_aware=True, profile="common", schema_spec=None):
        calls.append(
            {
                "seed": seed,
                "type_aware": type_aware,
                "profile": profile,
                "schema_spec": schema_spec,
            }
        )
        return _case(seed, profile=profile)

    batch = generate_candidate_batch(
        seed_start=10,
        candidate_pool=2,
        config=ExperimentConfig(enable_replay_bug=True),
        feedback=None,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=("context",),
        target_capabilities=("op:select",),
        schema_spec_for_seed=lambda seed: {"seed": seed},
        generate_case_fn=fake_generate_case,
        replay_filter_fn=lambda case, config: "",
    )

    assert [case.seed for case in batch.candidates] == [10, 11]
    assert batch.next_seed == 12
    assert batch.counter_deltas() == {
        "replay_filtered_candidates": 0,
        "replay_fallback_candidates": 0,
        "saturated_family_filtered_candidates": 0,
        "saturated_family_fallback_candidates": 0,
    }
    assert [call["schema_spec"] for call in calls] == [{"seed": 10}, {"seed": 11}]
    first_meta = batch.candidate_meta[id(batch.candidates[0])]
    assert first_meta["source"] == "generated"
    assert first_meta["generated_seed"] == 10
    assert first_meta["seed_lineage"]["depth"] == 0
    assert first_meta["generator_profile_selection"]["profile"] == "common"
    assert first_meta["replay_filter"]["enabled"] is False
    assert first_meta["family_saturation_filter"]["enabled"] is False


def test_generate_candidate_batch_uses_feedback_metadata_and_quality_context():
    class FakeFeedback:
        def __init__(self):
            self.last_candidate_source = "feedback_mutation"
            self.last_candidate_metadata = {}
            self.quality_target_keys = None

        def select_case(self, seed, generated):
            self.last_candidate_metadata = {
                "seed_lineage": {
                    "root_seed": generated.seed,
                    "parent_seed": 1,
                    "parent_case_id": "case-parent",
                    "mutation_seed": seed,
                    "depth": 1,
                },
                "mutation": {"operator": "value", "detail": "x", "changed": True},
                "feedback_selection": {"target_keys": ["semantic_family:cast_semantics"]},
            }
            return generated

        def candidate_quality_context(self, case, *, target_keys=None):
            self.quality_target_keys = list(target_keys or [])
            return {"archive_known": True, "target_keys": list(target_keys or [])}

    feedback = FakeFeedback()

    batch = generate_candidate_batch(
        seed_start=20,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=True),
        feedback=feedback,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=("op:select",),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=lambda case, config: "",
    )

    meta = batch.candidate_meta[id(batch.candidates[0])]
    assert meta["source"] == "feedback_mutation"
    assert meta["seed_lineage"]["parent_case_id"] == "case-parent"
    assert meta["mutation"]["operator"] == "value"
    assert meta["quality_archive_context"]["archive_known"] is True
    assert "semantic_family:cast_semantics" in feedback.quality_target_keys
    assert "capability:op:select" in feedback.quality_target_keys
    assert batch.candidates[0].metadata["quality_archive_context"] == meta["quality_archive_context"]


def test_generate_candidate_batch_consumes_feedback_seed_energy_batch():
    class BatchFeedback:
        def __init__(self):
            self.last_candidate_source = "generated"
            self.last_candidate_metadata = {}
            self.pending = []
            self.batch_calls = []

        def select_case_batch(self, seed, generated, *, max_batch=None, enqueue_remaining=False):
            self.batch_calls.append(
                {
                    "seed": seed,
                    "max_batch": max_batch,
                    "enqueue_remaining": enqueue_remaining,
                }
            )
            if self.pending:
                selected, metadata = self.pending.pop(0)
                self.last_candidate_source = metadata["source"]
                self.last_candidate_metadata = metadata
                return [selected]
            batch = []
            for offset in range(min(3, int(max_batch or 1))):
                candidate = _case(seed + offset + 100)
                metadata = {
                    "source": "feedback_mutation",
                    "seed_lineage": {
                        "root_seed": generated.seed,
                        "parent_index": 0,
                        "parent_case_id": "case-parent",
                        "mutation_seed": seed + offset,
                        "depth": 1,
                    },
                    "mutation": {
                        "operator": "value",
                        "detail": f"value:{offset}",
                        "changed": True,
                    },
                    "feedback_decision": {
                        "parent_index": 0,
                        "selected_operator": "value",
                    },
                }
                batch.append((candidate, metadata))
            self.pending.extend(batch[1:] if enqueue_remaining else [])
            selected, metadata = batch[0]
            self.last_candidate_source = metadata["source"]
            self.last_candidate_metadata = metadata
            return [candidate for candidate, _ in batch]

        def pop_pending_candidate(self):
            if not self.pending:
                return None
            selected, metadata = self.pending.pop(0)
            self.last_candidate_source = metadata["source"]
            self.last_candidate_metadata = metadata
            return SimpleNamespace(case=selected, source=metadata["source"], metadata=metadata)

        def candidate_quality_context(self, case, *, target_keys=None):
            return {}

    feedback = BatchFeedback()
    generate_calls = []

    def fake_generate_case(seed, **kwargs):
        generate_calls.append(seed)
        return _case(seed, profile=kwargs.get("profile", "common"))

    batch = generate_candidate_batch(
        seed_start=30,
        candidate_pool=3,
        config=ExperimentConfig(enable_replay_bug=True),
        feedback=feedback,
        guidance=None,
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=fake_generate_case,
        replay_filter_fn=lambda case, config: "",
    )

    assert generate_calls == [30]
    assert [case.case_id for case in batch.candidates] == [
        "case-130-common",
        "case-131-common",
        "case-132-common",
    ]
    assert batch.next_seed == 33
    assert [batch.candidate_meta[id(case)]["source"] for case in batch.candidates] == [
        "feedback_mutation",
        "feedback_mutation",
        "feedback_mutation",
    ]
    assert batch.candidate_meta[id(batch.candidates[0])]["seed_lineage"]["mutation_seed"] == 30
    assert batch.candidate_meta[id(batch.candidates[1])]["seed_lineage"]["mutation_seed"] == 31
    assert batch.candidate_meta[id(batch.candidates[2])]["seed_lineage"]["mutation_seed"] == 32
    assert feedback.batch_calls[0] == {
        "seed": 30,
        "max_batch": 3,
        "enqueue_remaining": True,
    }


def test_generate_candidate_batch_falls_back_after_replay_filter_saturation():
    def replay_filter(case, config):
        return "issue_replay_probe" if case.seed < 22 else ""

    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=False),
        feedback=None,
        guidance=None,
        generator_profile_pool=("known_replay_profile",),
        generator_profile_pool_metadata={"selected": ["known_replay_profile"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=replay_filter,
    )

    candidate = batch.candidates[0]
    meta = batch.candidate_meta[id(candidate)]
    assert candidate.seed == 22
    assert batch.next_seed == 23
    assert batch.replay_filtered_candidates == 21
    assert batch.replay_fallback_candidates == 1
    assert meta["source"] == "generated_fresh_fallback"
    assert meta["generated_seed"] == 22
    assert meta["replay_filter"]["filtered_before_candidate"] == 21
    assert meta["replay_filter"]["fallback_used"] is True
    assert meta["replay_filter"]["last_skip_reason"] == "issue_replay_probe"


def test_replay_filter_runs_before_feedback_selection_for_generated_candidates():
    class FeedbackShouldNotBeCalled:
        def select_case(self, seed, generated):
            raise AssertionError("replay-filtered generated candidates must not enter feedback selection")

    def replay_filter(case, config):
        return "issue_replay_probe" if case.seed < 22 else ""

    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=False),
        feedback=FeedbackShouldNotBeCalled(),
        guidance=None,
        generator_profile_pool=("known_replay_profile",),
        generator_profile_pool_metadata={"selected": ["known_replay_profile"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=replay_filter,
    )

    meta = batch.candidate_meta[id(batch.candidates[0])]
    assert meta["source"] == "generated_fresh_fallback"
    assert meta["replay_filter"]["filtered_before_candidate"] == 21
    assert meta["replay_filter"]["fallback_used"] is True


def test_generate_candidate_batch_falls_back_after_family_saturation_filter():
    class SaturatedGuidance:
        def predicted_saturated_family_roots_for_candidate(self, case, *, include_known_families=True):
            return ["window_boundary"]

    batch = generate_candidate_batch(
        seed_start=1,
        candidate_pool=1,
        config=ExperimentConfig(enable_replay_bug=True),
        feedback=None,
        guidance=SaturatedGuidance(),
        generator_profile_pool=("common",),
        generator_profile_pool_metadata={"selected": ["common"]},
        generator_profile_context_features=(),
        target_capabilities=(),
        schema_spec_for_seed=lambda seed: None,
        generate_case_fn=lambda seed, **kwargs: _case(seed, profile=kwargs.get("profile", "common")),
        replay_filter_fn=lambda case, config: "",
    )

    candidate = batch.candidates[0]
    meta = batch.candidate_meta[id(candidate)]
    assert candidate.seed == 22
    assert batch.next_seed == 23
    assert batch.saturated_family_filtered_candidates == 21
    assert batch.saturated_family_fallback_candidates == 1
    assert meta["source"] == "generated_saturation_fallback"
    assert meta["family_saturation_filter"]["filtered_before_candidate"] == 21
    assert meta["family_saturation_filter"]["fallback_used"] is True
    assert meta["family_saturation_filter"]["last_skip_reason"] == "window_boundary"


def test_generate_case_with_optional_schema_falls_back_for_legacy_generators():
    calls: list[tuple[int, dict]] = []

    def legacy_generate_case(seed, *, type_aware=True, profile="common"):
        calls.append((seed, {"type_aware": type_aware, "profile": profile}))
        return _case(seed, profile=profile)

    case = _generate_case_with_optional_schema(
        30,
        type_aware=False,
        profile="common",
        schema_spec={"columns": []},
        generate_case_fn=legacy_generate_case,
    )

    assert case.seed == 30
    assert calls == [(30, {"type_aware": False, "profile": "common"})]


def test_runner_reexports_run_candidates_helpers_for_compatibility():
    assert runner_module._generate_case_with_optional_schema is _generate_case_with_optional_schema
    assert runner_module._known_replay_source_filter_reason is _known_replay_source_filter_reason
    assert runner_module._replay_bug_filter_reason is _replay_bug_filter_reason
