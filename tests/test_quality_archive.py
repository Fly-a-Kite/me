from datadiff.quality_archive import QualityDiversityArchive


def test_quality_archive_tracks_elites_by_utility_and_reward():
    archive = QualityDiversityArchive(max_elites_per_cluster=2)
    cluster = "profile=generic|targets=semantic_family_null|ops=filter"
    archive.record_seed(cluster, 0, 1.0)
    archive.record_seed(cluster, 1, 3.0)
    archive.record_seed(cluster, 2, 2.0)

    assert archive.elite_indexes(cluster) == [1, 2]
    assert archive.seed_elite_bonus(cluster, 1) > archive.seed_elite_bonus(cluster, 2)
    assert archive.seed_elite_bonus(cluster, 0) == 0.0

    archive.record_outcome(cluster, index=0, reward=5.0)
    archive.record_outcome(cluster, index=0, reward=5.0)

    assert 0 in archive.elite_indexes(cluster)
    assert archive.cluster_reward_signal(cluster) > 0.0


def test_quality_archive_health_penalty_uses_invalid_fallback_and_false_positive_rates():
    archive = QualityDiversityArchive()
    cluster = "profile=generic|targets=semantic_family_cast|ops=mutate"
    archive.record_seed(cluster, 0, 1.0)

    assert archive.cluster_health_penalty(cluster) == 0.0

    archive.record_outcome(
        cluster,
        index=0,
        reward=0.0,
        preflight_valid=False,
        fallback_used=True,
        false_positive=True,
    )

    assert archive.cluster_health_penalty(cluster) > 0.0


def test_quality_archive_retains_only_live_seed_indexes_and_round_trips():
    archive = QualityDiversityArchive(max_elites_per_cluster=1)
    first_cluster = "profile=a|targets=x|ops=filter"
    second_cluster = "profile=b|targets=y|ops=sort"
    archive.record_seed(first_cluster, 0, 1.0)
    archive.record_seed(first_cluster, 1, 2.0)
    archive.record_seed(second_cluster, 2, 3.0)

    archive.retain_seeds({first_cluster: {1}})

    assert archive.elite_indexes(first_cluster) == [1]
    assert archive.elite_indexes(second_cluster) == []

    restored = QualityDiversityArchive.from_state_dict(archive.to_state_dict())

    assert restored.max_elites_per_cluster == 1
    assert restored.elite_indexes(first_cluster) == [1]
    assert restored.elite_indexes(second_cluster) == []
