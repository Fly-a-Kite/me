from datadiff import runner as runner_module
from datadiff.config import ExperimentConfig
from datadiff.run_state import (
    _build_closed_loop_state,
    _inject_champion_corpus,
    _restore_closed_loop_state,
)


class FakeScheduler:
    restored_calls = []
    init_calls = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.restored = False
        FakeScheduler.init_calls.append(dict(kwargs))

    @classmethod
    def from_state_dict(cls, data, **kwargs):
        instance = cls(**kwargs)
        instance.restored = True
        instance.data = dict(data)
        cls.restored_calls.append({"data": dict(data), "kwargs": dict(kwargs)})
        return instance

    def to_state_dict(self):
        return {"restored": self.restored, "kwargs": dict(self.kwargs)}


class FakeFeedbackState:
    restored_calls = []
    init_calls = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        self.source_scheduler = kwargs.get("source_scheduler")
        self.champion_registry = kwargs.get("champion_registry")
        self.champion_version_id = kwargs.get("champion_version_id", "")
        FakeFeedbackState.init_calls.append(dict(kwargs))

    @classmethod
    def from_state_dict(cls, data, **kwargs):
        instance = cls(**kwargs)
        instance.data = dict(data)
        cls.restored_calls.append({"data": dict(data), "kwargs": dict(kwargs)})
        return instance

    def to_state_dict(self):
        return {
            "kwargs": dict(self.kwargs),
            "champion_version_id": self.champion_version_id,
        }


class FakeGuidanceState:
    restored_calls = []
    init_calls = []

    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        FakeGuidanceState.init_calls.append(dict(kwargs))

    @classmethod
    def from_state_dict(cls, data, **kwargs):
        instance = cls(**kwargs)
        instance.data = dict(data)
        cls.restored_calls.append({"data": dict(data), "kwargs": dict(kwargs)})
        return instance

    def to_state_dict(self):
        return {"kwargs": dict(self.kwargs)}


def _reset_fakes():
    FakeScheduler.restored_calls.clear()
    FakeScheduler.init_calls.clear()
    FakeFeedbackState.restored_calls.clear()
    FakeFeedbackState.init_calls.clear()
    FakeGuidanceState.restored_calls.clear()
    FakeGuidanceState.init_calls.clear()


def test_restore_closed_loop_state_uses_saved_scheduler_and_injected_state_classes():
    _reset_fakes()
    champion_registry = object()
    config = ExperimentConfig(
        method_arm="p5_quality_diversity",
        enable_local_source_scheduler=True,
        enable_operator_swarm=False,
        enable_ir_rewrite_mutations=False,
        enable_divergence_conditioned_mutations=False,
        enable_shrink_mutations=False,
        enable_hierarchical_archive=False,
        enable_bd_axis_bandit=False,
        enable_bayesian_exploration=False,
        enable_seed_energy_batch=False,
        enable_seed_energy_tier_bandit=False,
        enable_per_operator_energy=False,
        enable_lineage_rarity=False,
        enable_minhash_dedup=False,
        enable_disagreement_bd_axis=False,
        enable_champion_graft_donor_bandit=False,
        local_source_exploration_weight=0.25,
        guidance_targets=["groupby"],
        semantic_focus_families=["conditional_semantics"],
    )
    closed_loop_state = {
        "seen_signatures": ["b", "a"],
        "feedback": {
            "max_corpus": 8,
            "source_scheduler": {"total_pulls": 3, "min_feedback_share": 0.2},
        },
        "guidance": {"feature_counts": {"op:filter": 2}},
    }

    seen, signal_seen, feedback, guidance = _restore_closed_loop_state(
        closed_loop_state,
        config=config,
        backends=["pandas", "duckdb"],
        guidance_enabled=True,
        feedback_enabled=True,
        champion_registry=champion_registry,
        champion_version_id="old->new",
        feedback_state_cls=FakeFeedbackState,
        guidance_state_cls=FakeGuidanceState,
        source_scheduler_cls=FakeScheduler,
    )

    assert seen == {"a", "b"}
    assert signal_seen == {"a", "b"}
    assert feedback.champion_registry is champion_registry
    assert feedback.champion_version_id == "old->new"
    assert feedback.source_scheduler.restored is True
    assert FakeScheduler.restored_calls[0]["data"]["total_pulls"] == 3
    assert FakeScheduler.restored_calls[0]["kwargs"]["exploration_weight"] == 0.25
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["source_scheduler"] is feedback.source_scheduler
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["corpus_mode"] == "semantic_plan_qd"
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_quality_archive"] is True
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_operator_swarm"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_ir_rewrite_mutations"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_divergence_conditioned_mutations"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_shrink_mutations"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_hierarchical_archive"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_bd_axis_bandit"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_bayesian_exploration"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_seed_energy_batch"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_seed_energy_tier_bandit"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_per_operator_energy"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_lineage_rarity"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_minhash_dedup"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_disagreement_bd_axis"] is False
    assert FakeFeedbackState.restored_calls[0]["kwargs"]["enable_champion_graft_donor_bandit"] is False
    guidance_kwargs = FakeGuidanceState.restored_calls[0]["kwargs"]
    assert "groupby" in guidance_kwargs["targets"]
    assert "semantic_family:conditional_semantics" in guidance_kwargs["targets"]
    assert guidance_kwargs["active_backends"] == ["pandas", "duckdb"]


def test_restore_closed_loop_state_can_disable_feedback_and_guidance():
    config = ExperimentConfig(enable_local_source_scheduler=True)

    seen, signal_seen, feedback, guidance = _restore_closed_loop_state(
        {
            "seen_signatures": ["discovery-a"],
            "signal_seen_signatures": ["signal-a"],
            "feedback": {"source_scheduler": {"total_pulls": 10}},
            "guidance": {"feature_counts": {"x": 1}},
        },
        config=config,
        backends=["pandas"],
        guidance_enabled=False,
        feedback_enabled=False,
        feedback_state_cls=FakeFeedbackState,
        guidance_state_cls=FakeGuidanceState,
        source_scheduler_cls=FakeScheduler,
    )

    assert seen == {"discovery-a"}
    assert signal_seen == {"signal-a"}
    assert feedback is None
    assert guidance is None


def test_build_closed_loop_state_serializes_feedback_scheduler_and_guidance():
    scheduler = FakeScheduler(exploration_weight=0.5)
    feedback = FakeFeedbackState(source_scheduler=scheduler)
    guidance = FakeGuidanceState(targets=["groupby"])

    payload = _build_closed_loop_state(
        seen={"b", "a"},
        signal_seen={"signal-b", "signal-a"},
        feedback=feedback,
        guidance=guidance,
    )

    assert payload["seen_signatures"] == ["a", "b"]
    assert payload["signal_seen_signatures"] == ["signal-a", "signal-b"]
    assert payload["feedback"]["kwargs"]["source_scheduler"] is scheduler
    assert payload["source_scheduler"]["kwargs"]["exploration_weight"] == 0.5
    assert payload["guidance"]["kwargs"]["targets"] == ["groupby"]


def test_inject_champion_corpus_counts_accepted_champions():
    class FakeRegistry:
        def __init__(self):
            self.calls = []

        def champions_for_version(self, version_id, *, limit=None):
            self.calls.append({"version_id": version_id, "limit": limit})
            return ["accepted-a", "skip", "accepted-b"]

    class FakeFeedback:
        def __init__(self):
            self.champion_registry = FakeRegistry()
            self.injected = []

        def inject_champion_seed(self, champion):
            self.injected.append(champion)
            return champion != "skip"

    feedback = FakeFeedback()

    assert _inject_champion_corpus(feedback, version_id="new", limit=2) == 2
    assert feedback.champion_registry.calls == [{"version_id": "new", "limit": 2}]
    assert feedback.injected == ["accepted-a", "skip", "accepted-b"]
    assert _inject_champion_corpus(None, version_id="new", limit=2) == 0


def test_runner_reexports_run_state_helpers_for_compatibility():
    assert runner_module._restore_closed_loop_state is _restore_closed_loop_state
    assert runner_module._build_closed_loop_state is _build_closed_loop_state
    assert runner_module._inject_champion_corpus is _inject_champion_corpus
