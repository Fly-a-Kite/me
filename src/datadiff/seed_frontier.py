from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


SeedPriority = tuple[float, float, float, int, int, int]
SeedRowBuilder = Callable[[int], dict[str, Any]]
SeedPriorityBuilder = Callable[[int], SeedPriority]


@dataclass(slots=True)
class SeedFrontier:
    priority_builder: SeedPriorityBuilder
    row_builder: SeedRowBuilder
    heap: list[SeedPriority] = field(default_factory=list, repr=False)
    dirty: bool = True

    def mark_dirty(self) -> None:
        self.dirty = True

    def ensure(self, indexes: Iterable[int]) -> None:
        if not self.dirty:
            return
        heap: list[SeedPriority] = []
        for index in indexes:
            heapq.heappush(heap, self.priority_builder(int(index)))
        self.heap = heap
        self.dirty = False

    def snapshot(
        self,
        *,
        indexes: Iterable[int],
        limit: int,
        excluded_indexes: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure(indexes)
        excluded = excluded_indexes or set()
        rows: list[dict[str, Any]] = []
        for priority in heapq.nsmallest(max(0, int(limit)) + len(excluded), self.heap):
            index = int(priority[-1])
            if index in excluded:
                continue
            rows.append(self.row_builder(index))
            if len(rows) >= limit:
                break
        return rows

    def rank(self, *, indexes: Iterable[int], index: int) -> int:
        self.ensure(indexes)
        selected = next((priority for priority in self.heap if int(priority[-1]) == int(index)), None)
        if selected is None:
            return 0
        return 1 + sum(1 for priority in self.heap if priority < selected)
