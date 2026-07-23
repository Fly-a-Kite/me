from __future__ import annotations

from dataclasses import replace

import pytest

from datadiff_osc.parallel.dag import (
    TaskGraphError,
    build_task_graph,
    freeze_epoch,
    logical_task_digest,
    retry_task,
)
from datadiff_osc.schemas import (
    ResourceTokens,
    SeedLineage,
    SeedStage,
    TaskIdentity,
    TaskKind,
    TaskSpec,
)


def make_task(index, *, dependencies=(), epoch=3, resources=None):
    lineage = SeedLineage(
        protocol_digest="protocol-1",
        master_seed=31000001,
        lane_id="lane-a",
        case_index=index,
        stage_name=SeedStage.BACKEND,
    )
    identity = TaskIdentity(
        protocol_digest="protocol-1",
        task_kind=TaskKind.BACKEND_EXECUTION,
        epoch_index=epoch,
        decision_index=index,
        seed_lineage_digest=lineage.digest,
        endpoint_id=f"endpoint-{index}",
        backend="backend-a",
    )
    return TaskSpec(
        identity=identity,
        dependency_task_ids=tuple(dependencies),
        resources=resources
        or ResourceTokens(
            cpu_tokens=1,
            rss_bytes=16,
            io_class="default",
            backend_internal_threads=0,
        ),
        payload_digest=f"payload-{index}",
    )


def test_task_graph_is_stable_across_input_order_and_forms_layers():
    first = make_task(0)
    second = make_task(1, dependencies=(first.identity.task_id,))
    third = make_task(2, dependencies=(first.identity.task_id,))
    left = build_task_graph((third, first, second))
    right = build_task_graph((second, third, first))
    assert left.digest == right.digest
    assert left.layers == (
        (first.identity.task_id,),
        tuple(sorted((second.identity.task_id, third.identity.task_id))),
    )


def test_task_graph_rejects_missing_dependency_and_cycle():
    first = make_task(0)
    with pytest.raises(TaskGraphError, match="unknown dependencies"):
        build_task_graph((replace(first, dependency_task_ids=("missing",)),))

    second = make_task(1)
    cyclic_first = replace(first, dependency_task_ids=(second.identity.task_id,))
    cyclic_second = replace(second, dependency_task_ids=(first.identity.task_id,))
    with pytest.raises(TaskGraphError, match="cycle"):
        build_task_graph((cyclic_first, cyclic_second))


def test_retry_changes_only_attempt_and_preserves_logical_task_and_seed():
    task = make_task(4)
    retried = retry_task(task)
    assert retried.identity.task_id != task.identity.task_id
    assert retried.identity.attempt == task.identity.attempt + 1
    assert retried.identity.seed_lineage_digest == task.identity.seed_lineage_digest
    assert retried.payload_digest == task.payload_digest
    assert retried.resources == task.resources
    assert logical_task_digest(retried) == logical_task_digest(task)


def test_frozen_epoch_rejects_mixed_epoch_and_binds_assignment():
    first = make_task(0)
    epoch = freeze_epoch((first,), assignment_digest="assignment-1")
    assert epoch.epoch_index == 3
    assert epoch.assignment_digest == "assignment-1"
    assert epoch.digest == freeze_epoch((first,), assignment_digest="assignment-1").digest
    with pytest.raises(TaskGraphError, match="mix protocol or epoch"):
        freeze_epoch((first, make_task(1, epoch=4)), assignment_digest="assignment-1")
