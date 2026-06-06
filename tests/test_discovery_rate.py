from datadiff.discovery_rate import DiscoveryRateEstimator


def test_good_turing_unseen_probability_tracks_singletons():
    estimator = DiscoveryRateEstimator()
    for index in range(100):
        estimator.observe([f"family:{index}"])

    assert estimator.unseen_probability() == 1.0
    assert estimator.bucket() == "very_high"
    high_weight = estimator.adaptive_exploration_weight(0.75)

    for _ in range(100):
        estimator.observe(["family:0"])

    assert estimator.unseen_probability() < 0.50
    assert estimator.adaptive_exploration_weight(0.75) < high_weight


def test_discovery_rate_state_round_trips_counts():
    estimator = DiscoveryRateEstimator()
    estimator.observe(["family:a", "family:b"])
    estimator.observe(["family:a"])

    restored = DiscoveryRateEstimator.from_state_dict(estimator.to_state_dict())

    assert restored.family_counts == estimator.family_counts
    assert restored.total_observations == 3
    assert restored.singleton_family_count == 1
