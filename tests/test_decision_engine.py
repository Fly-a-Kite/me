from types import SimpleNamespace

from datadiff.decision_engine import (
    choose_priority_actions,
    choose_generator_profile,
    record_adaptive_action_feedback,
    record_priority_action_feedback,
)


def test_choose_generator_profile_empty_pool_defaults_to_common():
    profile, selection = choose_generator_profile(
        None,
        (),
        context_features=(),
        learning_weight=1.0,
        pool_metadata={"capability_aware": True},
    )

    assert profile == "common"
    assert selection == {
        "strategy": "fixed",
        "profile": "common",
        "profile_pool": [],
        "profile_pool_metadata": {"capability_aware": True},
        "learning_weight": 0.0,
        "ranked": [],
    }


def test_record_adaptive_action_feedback_records_contextual_bandit_selection():
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeLearning:
        def record_outcome(self, *args, **kwargs):
            calls.append((args, kwargs))

    reward = record_adaptive_action_feedback(
        SimpleNamespace(adaptive_learning=FakeLearning()),
        {
            "strategy": "contextual_bandit",
            "scope": "semantic_objective",
            "action": "objective_a",
        },
        context_features=("family:agg",),
        version_id="latest->fixed",
        reward=1.75,
        runtime_cost=0.25,
        preflight_valid=False,
        fallback_used=True,
        false_positive=True,
    )

    assert reward == 1.75
    assert calls == [
        (
            ("semantic_objective", "objective_a"),
            {
                "context_features": ("family:agg",),
                "version_id": "latest->fixed",
                "reward": 1.75,
                "runtime_cost": 0.25,
                "preflight_valid": False,
                "fallback_used": True,
                "false_positive": True,
            },
        )
    ]


def test_record_adaptive_action_feedback_ignores_fixed_selection():
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeLearning:
        def record_outcome(self, *args, **kwargs):
            calls.append((args, kwargs))

    reward = record_adaptive_action_feedback(
        SimpleNamespace(adaptive_learning=FakeLearning()),
        {
            "strategy": "fixed",
            "scope": "semantic_objective",
            "action": "objective_a",
        },
        context_features=("family:agg",),
        version_id="latest->fixed",
        reward=1.75,
    )

    assert reward is None
    assert calls == []


def test_choose_priority_actions_uses_warmup_then_ranked_priority():
    class Arm:
        def __init__(self, pulls):
            self.pulls = pulls

    class FakeLearning:
        def __init__(self, pulls):
            self.bandits = {
                "backend_pair": SimpleNamespace(
                    arms={action: Arm(pull_count) for action, pull_count in pulls.items()}
                )
            }

        def rank_top(self, scope, action_ids, *, limit, **kwargs):
            assert scope == "backend_pair"
            return [
                {"action_id": "b|c", "score": 3.0},
                {"action_id": "a|c", "score": 2.0},
                {"action_id": "a|b", "score": 1.0},
            ][:limit]

    pool = ("a|b", "a|c", "b|c")
    warmup = choose_priority_actions(
        SimpleNamespace(adaptive_learning=FakeLearning({"a|b": 1, "a|c": 0, "b|c": 0})),
        scope="backend_pair",
        action_pool=pool,
        context_features=("root:join",),
        version_id="latest",
        learning_weight=1.0,
        enabled=True,
        limit=2,
    )
    ranked = choose_priority_actions(
        SimpleNamespace(adaptive_learning=FakeLearning({"a|b": 1, "a|c": 1, "b|c": 1})),
        scope="backend_pair",
        action_pool=pool,
        context_features=("root:join",),
        version_id="latest",
        learning_weight=1.0,
        enabled=True,
        limit=2,
    )

    assert warmup.strategy == "contextual_bandit_warmup"
    assert warmup.priority == ("a|c", "b|c")
    assert ranked.strategy == "contextual_bandit"
    assert ranked.priority == ("b|c", "a|c")
    assert ranked.to_selection_dict()["priority"] == ["b|c", "a|c"]


def test_choose_priority_actions_fixed_when_disabled():
    decision = choose_priority_actions(
        None,
        scope="backend_pair",
        action_pool=("a|b", "a|c"),
        context_features=(),
        version_id="",
        learning_weight=1.0,
        enabled=False,
        limit=1,
    )

    assert decision.strategy == "fixed"
    assert decision.priority == ("a|b",)
    assert decision.learning_weight == 0.0


def test_record_priority_action_feedback_records_each_priority_action():
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FakeLearning:
        def record_outcome(self, *args, **kwargs):
            calls.append((args, kwargs))

    reward = record_priority_action_feedback(
        SimpleNamespace(adaptive_learning=FakeLearning()),
        {
            "strategy": "contextual_bandit",
            "scope": "backend_pair",
            "priority": ["a|b"],
        },
        "a|b",
        context_features=("root:join",),
        version_id="latest",
        reward=2.5,
        runtime_cost=0.1,
    )

    assert reward == 2.5
    assert calls == [
        (
            ("backend_pair", "a|b"),
            {
                "context_features": ("root:join",),
                "version_id": "latest",
                "reward": 2.5,
                "runtime_cost": 0.1,
                "preflight_valid": True,
                "fallback_used": False,
                "false_positive": False,
            },
        )
    ]
