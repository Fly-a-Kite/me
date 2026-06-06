import random

from datadiff.mutator_swarm import OperatorSwarm


def test_operator_swarm_initializes_normalized_particles():
    swarm = OperatorSwarm.init(["value", "drop_row", "append_range_filter"], n_particles=4, seed=7)

    assert len(swarm.particles) == 4
    assert swarm.operator_names == ("value", "drop_row", "append_range_filter")
    for particle in swarm.particles:
        assert set(particle.weights) == set(swarm.operator_names)
        assert abs(sum(particle.weights.values()) - 1.0) < 1e-9


def test_operator_swarm_positive_feedback_increases_operator_weight_and_bonus():
    swarm = OperatorSwarm.init(["value", "drop_row", "append_range_filter"], n_particles=2, seed=11)
    particle_id = 0
    before = swarm.select_particle(particle_id).weights["append_range_filter"]

    for step in range(24):
        swarm.update(
            particle_id,
            3.0,
            operator="append_range_filter",
            rnd=random.Random(step),
        )

    after = swarm.select_particle(particle_id).weights["append_range_filter"]

    assert after > before
    assert swarm.operator_pulls["append_range_filter"] == 24
    assert swarm.score_bonus("append_range_filter", particle_id=particle_id) > 0.0


def test_operator_swarm_state_round_trips_weights_and_stats():
    swarm = OperatorSwarm.init(["value", "drop_row"], n_particles=3, seed=13)
    swarm.update(1, 2.0, operator="drop_row", rnd=random.Random(1))

    restored = OperatorSwarm.from_state_dict(swarm.to_state_dict(), operator_names=["value", "drop_row"])

    assert restored.operator_names == swarm.operator_names
    assert restored.operator_pulls == swarm.operator_pulls
    assert restored.operator_rewards == swarm.operator_rewards
    assert restored.select_particle(1).pulls == 1
    assert restored.select_particle(1).weights == swarm.select_particle(1).weights


def test_operator_swarm_legacy_empty_state_falls_back_to_initialized_pool():
    swarm = OperatorSwarm.from_state_dict(None, operator_names=["value", "drop_row"])

    assert len(swarm.particles) == 16
    assert set(swarm.particles[0].weights) == {"value", "drop_row"}
