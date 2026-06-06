from datadiff.quality_archive import QualityDiversityArchive


def _descriptor() -> dict[str, str]:
    return {
        "profile_axis": "common",
        "op_skeleton_axis": "op:filter_groupby",
        "target_class_axis": "target:semantic_family_aggregation",
        "null_density_axis": "null:1",
        "type_mix_axis": "type:num1_str1",
        "row_mass_axis": "rows:2",
        "column_count_axis": "cols:2",
        "backend_disagreement_axis": "pair:duckdb_pandas",
    }


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


def test_quality_archive_records_multi_axis_cells_and_composite_reward():
    archive = QualityDiversityArchive(max_elites_per_cluster=2)
    cluster = "profile=common|targets=semantic_family_aggregation|ops=filter_groupby"
    descriptor = _descriptor()

    archive.record_seed_multi(cluster, descriptor, 0, 2.0)
    archive.record_outcome_multi(cluster, descriptor, index=0, reward=3.0)
    archive.record_outcome_multi(cluster, descriptor, index=0, reward=3.0)

    assert archive.elite_indexes(cluster) == [0]
    assert archive.axis_seed_count("bd_op_skeleton", "op:filter_groupby") == 1
    assert archive.axis_seed_count("bd_backend_disagreement", "pair:duckdb_pandas") == 1
    assert archive.composite_reward_signal(descriptor) > 0.0

    restored = QualityDiversityArchive.from_state_dict(archive.to_state_dict())

    assert restored.axis_seed_count("bd_op_skeleton", "op:filter_groupby") == 1
    assert restored.axis_seed_count("bd_backend_disagreement", "pair:duckdb_pandas") == 1
    assert restored.composite_reward_signal(descriptor) == archive.composite_reward_signal(descriptor)


def test_quality_archive_splits_high_variance_cells_by_descriptor_axis_and_round_trips():
    archive = QualityDiversityArchive(
        max_elites_per_cluster=2,
        split_min_seeds=4,
        split_variance_threshold=0.05,
        merge_variance_threshold=0.0,
    )
    cluster = "profile=common|targets=semantic_family_aggregation|ops=filter_groupby"
    duckdb_descriptor = dict(_descriptor())
    pandas_descriptor = {**_descriptor(), "backend_disagreement_axis": "pair:pandas_sqlite"}

    archive.record_seed_multi(cluster, duckdb_descriptor, 0, 0.0)
    archive.record_seed_multi(cluster, duckdb_descriptor, 1, 0.1)
    archive.record_seed_multi(cluster, pandas_descriptor, 2, 3.0)
    archive.record_seed_multi(cluster, pandas_descriptor, 3, 3.2)

    assert archive.split_cell_count() == 1
    assert archive.hierarchical_cell_count(cluster) == 2
    assert archive.seed_elite_bonus(cluster, 3) > 0.0
    assert archive.elite_indexes(cluster)

    restored = QualityDiversityArchive.from_state_dict(archive.to_state_dict())

    assert restored.split_cell_count() == 1
    assert restored.hierarchical_cell_count(cluster) == 2
    assert restored.elite_indexes(cluster) == archive.elite_indexes(cluster)


def test_quality_archive_merges_low_variance_hierarchical_siblings():
    archive = QualityDiversityArchive(
        split_min_seeds=4,
        split_variance_threshold=0.05,
        merge_variance_threshold=0.25,
    )
    cluster = "profile=common|targets=semantic_family_aggregation|ops=filter_groupby"
    first_descriptor = dict(_descriptor())
    second_descriptor = {**_descriptor(), "backend_disagreement_axis": "pair:pandas_sqlite"}

    archive.record_seed_multi(cluster, first_descriptor, 0, 0.0)
    archive.record_seed_multi(cluster, first_descriptor, 1, 0.1)
    archive.record_seed_multi(cluster, second_descriptor, 2, 3.0)
    archive.record_seed_multi(cluster, second_descriptor, 3, 3.2)
    assert archive.hierarchical_cell_count(cluster) == 2

    for _ in range(3):
        archive.record_outcome_multi(cluster, first_descriptor, index=0, reward=0.1)
        archive.record_outcome_multi(cluster, second_descriptor, index=2, reward=0.1)

    assert archive.split_cell_count() == 0
    assert archive.hierarchical_cell_count(cluster) == 0


def test_quality_archive_can_disable_hierarchical_split_merge():
    archive = QualityDiversityArchive(
        enable_hierarchical=False,
        split_min_seeds=4,
        split_variance_threshold=0.05,
    )
    cluster = "profile=common|targets=semantic_family_aggregation|ops=filter_groupby"
    descriptor = _descriptor()

    for index, utility in enumerate((0.0, 0.1, 3.0, 3.2)):
        archive.record_seed_multi(cluster, descriptor, index, utility)

    assert archive.maybe_split_cell(cluster) is False
    assert archive.split_cell_count() == 0
    assert archive.hierarchical_cell_count(cluster) == 0


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
