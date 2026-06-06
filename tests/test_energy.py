from datadiff.energy import EnergyParameters, cost_normalized_reward, operator_energy, seed_energy


def test_seed_energy_rewards_new_productive_diverse_seeds():
    baseline = seed_energy(
        pulls=8,
        mean_reward=0.1,
        family_breadth=1,
        cluster_outcome_count=0,
        since_last_finding_pulls=8,
    )
    rich = seed_energy(
        pulls=0,
        mean_reward=2.0,
        family_breadth=5,
        cluster_outcome_count=4,
        since_last_finding_pulls=0,
    )

    assert rich > baseline


def test_seed_energy_is_bounded_by_parameters():
    params = EnergyParameters(min_energy=2, max_energy=5)

    assert seed_energy(
        pulls=0,
        mean_reward=100.0,
        family_breadth=100,
        cluster_outcome_count=100,
        since_last_finding_pulls=0,
        params=params,
    ) == 5
    assert seed_energy(
        pulls=100,
        mean_reward=-100.0,
        family_breadth=0,
        cluster_outcome_count=0,
        since_last_finding_pulls=100,
        params=params,
    ) == 2


def test_operator_energy_cools_repeated_unproductive_operators():
    productive = operator_energy(
        pulls=1,
        mean_reward=2.0,
        recent_unproductive_streak=0,
        catalog_width=32,
    )
    stale = operator_energy(
        pulls=20,
        mean_reward=0.0,
        recent_unproductive_streak=8,
        catalog_width=32,
    )

    assert productive > stale


def test_cost_normalized_reward_prefers_same_signal_at_lower_elapsed_cost():
    fast = cost_normalized_reward(4.0, 0.5)
    slow = cost_normalized_reward(4.0, 2.0)

    assert fast > slow
    assert cost_normalized_reward(4.0, 0.0) == 80.0
