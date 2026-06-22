from datadiff.seed_frontier import SeedFrontier


def test_seed_frontier_snapshot_and_rank_follow_priority_order():
    rows = {
        0: {"index": 0, "case_id": "a", "schedule_score": 2.0},
        1: {"index": 1, "case_id": "b", "schedule_score": 3.0},
        2: {"index": 2, "case_id": "c", "schedule_score": 1.0},
    }

    frontier = SeedFrontier(
        priority_builder=lambda index: (-rows[index]["schedule_score"], 0.0, 0.0, 0, 0, index),
        row_builder=lambda index: dict(rows[index]),
    )

    snapshot = frontier.snapshot(indexes=range(3), limit=2)

    assert [row["case_id"] for row in snapshot] == ["b", "a"]
    assert frontier.rank(indexes=range(3), index=1) == 1
    assert frontier.rank(indexes=range(3), index=0) == 2


def test_seed_frontier_mark_dirty_rebuilds_heap():
    priorities = {0: 1.0, 1: 2.0}
    frontier = SeedFrontier(
        priority_builder=lambda index: (-priorities[index], 0.0, 0.0, 0, 0, index),
        row_builder=lambda index: {"index": index, "schedule_score": priorities[index]},
    )

    assert frontier.snapshot(indexes=range(2), limit=1)[0]["index"] == 1
    priorities[0] = 5.0
    frontier.mark_dirty()

    assert frontier.snapshot(indexes=range(2), limit=1)[0]["index"] == 0
