from __future__ import annotations

from datadiff_osc.schemas import SeedLineage, SeedStage
from datadiff_osc.search.epochs import NoReplacementEpoch, derive_stage_lineage


def _base():
    return SeedLineage("protocol", 31000001, "lane-3", 0, SeedStage.TARGET)


def test_epoch_is_no_replacement_and_reproducible():
    items = tuple(f"cell-{index:03d}" for index in range(232))
    left = NoReplacementEpoch(items, _base(), namespace="cells")
    right = NoReplacementEpoch(reversed(items), _base(), namespace="cells")
    first = tuple(left.select(index).item_id for index in range(232))
    assert len(first) == len(set(first)) == 232
    assert first == tuple(right.select(index).item_id for index in range(232))
    assert set(first) == set(items)


def test_worker_partition_completion_order_and_retry_do_not_change_selection():
    epoch = NoReplacementEpoch((f"edge-{i}" for i in range(384)), _base(), namespace="edges")
    serial = {index: epoch.select(index) for index in range(768)}
    completion_order = list(range(767, -1, -1))
    parallel = {index: epoch.select(index) for index in completion_order}
    assert [(i, serial[i].item_id, serial[i].seed_lineage.digest) for i in serial] == [
        (i, parallel[i].item_id, parallel[i].seed_lineage.digest) for i in serial
    ]
    assert epoch.select(17) == epoch.select(17)


def test_stage_substreams_are_keyed_and_isolated():
    data = derive_stage_lineage(_base(), SeedStage.DATA, counter=2)
    mutation = derive_stage_lineage(_base(), SeedStage.MUTATION, counter=2)
    assert data.subseed != mutation.subseed
    assert data.parent_digest == mutation.parent_digest == _base().digest
