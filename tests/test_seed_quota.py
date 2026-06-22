from datadiff.seed_quota import SeedEvictionPolicy


def test_seed_quota_evicts_from_overfull_incoming_cluster_before_rare_singleton():
    quota = SeedEvictionPolicy()

    index = quota.evict_candidate(
        case_cluster_keys=["common", "rare", "common"],
        case_utilities=[1.0, 1.0, 1.4],
        case_mutation_pulls=[0, 0, 0],
        incoming_cluster_key="common",
        incoming_utility=3.0,
        max_corpus=3,
    )

    assert index == 0


def test_seed_quota_uses_global_weakest_when_capacity_cannot_cover_all_cells():
    quota = SeedEvictionPolicy()

    index = quota.evict_candidate(
        case_cluster_keys=["a", "b", "c"],
        case_utilities=[2.0, 1.0, 3.0],
        case_mutation_pulls=[0, 0, 0],
        incoming_cluster_key="d",
        incoming_utility=4.0,
        max_corpus=3,
    )

    assert index == 1


def test_seed_quota_round_trips_config():
    quota = SeedEvictionPolicy(enabled=False, min_quota_per_active_cell=2, pull_decay=0.5)

    restored = SeedEvictionPolicy.from_state_dict(quota.to_state_dict())

    assert restored.enabled is False
    assert restored.min_quota_per_active_cell == 2
    assert restored.pull_decay == 0.5
