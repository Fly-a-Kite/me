from datadiff.lineage import LineageDAG


def test_lineage_rarity_cools_crowded_successful_sibling_branch():
    dag = LineageDAG()
    dag.add_seed(0)
    dag.add_seed(1, parent_index=0)
    dag.add_seed(2, parent_index=0)

    before = dag.rarity_score(2)
    for _ in range(8):
        dag.record_pull(1, reward=1.0, family_keys=["family:a"])

    assert dag.rarity_score(2) < before
    assert dag.rarity_score(1) < 1.0


def test_lineage_state_round_trips_parent_child_links():
    dag = LineageDAG()
    dag.add_seed(0)
    dag.add_seed(1, parent_index=0)
    dag.record_pull(1, reward=2.0, family_keys=["family:a"])

    restored = LineageDAG.from_state_dict(dag.to_state_dict())

    assert restored.nodes[1].parent_index == 0
    assert 1 in restored.nodes[0].children
    assert restored.nodes[1].pulls == 1
    assert restored.nodes[1].found_unique_families == {"family:a"}
