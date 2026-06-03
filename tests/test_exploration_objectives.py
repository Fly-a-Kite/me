from datadiff.exploration_objectives import (
    ExplorationObjectiveRule,
    clear_runtime_exploration_objective_rules,
    derive_exploration_objective_features,
    objective_features_for_rules,
    register_exploration_objective_rule,
)


def test_exploration_objective_rules_derive_default_and_runtime_features():
    clear_runtime_exploration_objective_rules()
    try:
        assert "exploration_objective:predicate_logic" in derive_exploration_objective_features(
            ["op:filter", "filter:truth-test"]
        )

        register_exploration_objective_rule(
            {
                "objective": "custom breadth",
                "exact_features": ["feature:dynamic"],
            }
        )

        assert "exploration_objective:custom_breadth" in derive_exploration_objective_features(
            ["feature:dynamic"]
        )
    finally:
        clear_runtime_exploration_objective_rules()


def test_objective_features_for_rules_normalizes_objective_names():
    rules = [
        ExplorationObjectiveRule("objective:adaptive lane", exact_features=frozenset({"op:join"})),
        {"objective": "exploration_objective:adaptive lane", "fragments": ["join"]},
    ]

    assert objective_features_for_rules(rules) == ["exploration_objective:adaptive_lane"]
