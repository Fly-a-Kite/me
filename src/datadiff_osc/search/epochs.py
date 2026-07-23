"""Worker-independent keyed substreams and no-replacement epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from datadiff_osc._canonical import stable_digest
from datadiff_osc.schemas import SeedLineage, SeedStage


def derive_stage_lineage(
    parent: SeedLineage,
    stage: SeedStage,
    *,
    case_index: int | None = None,
    counter: int = 0,
) -> SeedLineage:
    return SeedLineage(
        protocol_digest=parent.protocol_digest,
        master_seed=parent.master_seed,
        lane_id=parent.lane_id,
        case_index=parent.case_index if case_index is None else int(case_index),
        stage_name=stage,
        counter=int(counter),
        parent_digest=parent.digest,
    )


@dataclass(frozen=True, slots=True)
class EpochSelection:
    item_id: str
    epoch_index: int
    position: int
    decision_index: int
    seed_lineage: SeedLineage


class NoReplacementEpoch:
    def __init__(
        self,
        item_ids: Iterable[str],
        lineage: SeedLineage,
        *,
        namespace: str,
    ) -> None:
        materialized = tuple(sorted(str(item) for item in item_ids))
        if not materialized or len(materialized) != len(set(materialized)):
            raise ValueError("epoch items must be non-empty and unique")
        if not namespace:
            raise ValueError("epoch namespace must be non-empty")
        self.item_ids = materialized
        self.lineage = lineage
        self.namespace = namespace

    def permutation(self, epoch_index: int) -> tuple[str, ...]:
        if epoch_index < 0:
            raise ValueError("epoch index must be non-negative")
        identity = {
            "protocol": self.lineage.protocol_digest,
            "master_seed": self.lineage.master_seed,
            "lane_id": self.lineage.lane_id,
            "namespace": self.namespace,
            "epoch_index": int(epoch_index),
        }
        return tuple(
            sorted(
                self.item_ids,
                key=lambda item: (
                    stable_digest("osc-no-replacement-rank", {**identity, "item_id": item}),
                    item,
                ),
            )
        )

    def select(self, decision_index: int) -> EpochSelection:
        if decision_index < 0:
            raise ValueError("decision index must be non-negative")
        epoch_index, position = divmod(int(decision_index), len(self.item_ids))
        item_id = self.permutation(epoch_index)[position]
        return EpochSelection(
            item_id=item_id,
            epoch_index=epoch_index,
            position=position,
            decision_index=int(decision_index),
            seed_lineage=derive_stage_lineage(
                self.lineage,
                self.lineage.stage_name,
                case_index=int(decision_index),
                counter=epoch_index,
            ),
        )


__all__ = ["EpochSelection", "NoReplacementEpoch", "derive_stage_lineage"]
