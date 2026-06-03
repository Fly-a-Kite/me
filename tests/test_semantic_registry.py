from datadiff.exploration_objectives import ExplorationObjectiveRule
from datadiff.mutator import mutation_operator_profiles
from datadiff.semantic_registry import (
    filter_generator_profiles_by_capability,
    generator_profile_capability_requirements,
    semantic_registry_payload,
)


def test_semantic_registry_exposes_objective_capability_oracle_template():
    payload = semantic_registry_payload(
        objective_rules=[
            ExplorationObjectiveRule(
                "adaptive_consistency",
                exact_features=frozenset({"op:join"}),
            )
        ],
        target_context={"common_capabilities": ["op:join", "op:filter"]},
        operator_profiles=mutation_operator_profiles(allow_probe_operators=False),
    )

    assert payload["schema_version"] == "semantic-registry-v1"
    assert payload["target_capabilities"] == ["op:join", "op:filter"]
    objective = payload["objectives"][0]
    assert objective["feature"] == "exploration_objective:adaptive_consistency"
    assert "cross_version" in objective["oracle_roles"]
    assert payload["generator_profiles"][0]["profile"] == "common"
    assert payload["extension_steps"][-1] == "triage_reduce_recheck_and_record_ledger"


def test_generator_profile_capability_filter_drops_unsupported_profiles():
    filtered = filter_generator_profiles_by_capability(
        ["common", "join_null_key_topk", "partitioned_running_sum"],
        ["table:single", "op:filter", "op:join", "op:sort", "op:limit"],
    )

    assert "op:join" in generator_profile_capability_requirements("join_null_key_topk")
    assert filtered["selected"] == ["common", "join_null_key_topk"]
    assert filtered["dropped"][0]["profile"] == "partitioned_running_sum"
    assert "op:running_sum" in filtered["dropped"][0]["missing"]
