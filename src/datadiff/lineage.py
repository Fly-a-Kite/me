from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

LINEAGE_SCHEMA_VERSION = "lineage-dag-v1"


@dataclass(slots=True)
class LineageNode:
    index: int
    parent_index: int | None = None
    children: set[int] = field(default_factory=set)
    reward_total: float = 0.0
    pulls: int = 0
    found_unique_families: set[str] = field(default_factory=set)

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "parent_index": self.parent_index,
            "children": sorted(self.children),
            "reward_total": float(self.reward_total),
            "pulls": int(self.pulls),
            "found_unique_families": sorted(self.found_unique_families),
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> "LineageNode":
        parent = data.get("parent_index", None)
        return cls(
            index=int(data.get("index", 0) or 0),
            parent_index=int(parent) if parent is not None else None,
            children={int(item) for item in data.get("children", []) or []},
            reward_total=float(data.get("reward_total", 0.0) or 0.0),
            pulls=int(data.get("pulls", 0) or 0),
            found_unique_families={
                str(item)
                for item in data.get("found_unique_families", []) or []
                if str(item)
            },
        )


@dataclass(slots=True)
class LineageDAG:
    nodes: dict[int, LineageNode] = field(default_factory=dict)
    roots: set[int] = field(default_factory=set)

    def add_seed(self, index: int, parent_index: int | None = None) -> None:
        node_index = int(index)
        parent = int(parent_index) if parent_index is not None and int(parent_index) >= 0 else None
        node = self.nodes.get(node_index)
        if node is None:
            node = LineageNode(index=node_index, parent_index=parent)
            self.nodes[node_index] = node
        else:
            self._unlink_from_parent(node_index, node.parent_index)
            node.parent_index = parent
        if parent is None or parent == node_index:
            node.parent_index = None
            self.roots.add(node_index)
            return
        self.roots.discard(node_index)
        parent_node = self.nodes.setdefault(parent, LineageNode(index=parent))
        parent_node.children.add(node_index)

    def remove_seed(self, index: int) -> None:
        node_index = int(index)
        node = self.nodes.pop(node_index, None)
        self.roots.discard(node_index)
        if node is None:
            return
        self._unlink_from_parent(node_index, node.parent_index)
        for child in list(node.children):
            child_node = self.nodes.get(child)
            if child_node is not None:
                child_node.parent_index = None
                self.roots.add(child)

    def record_pull(
        self,
        index: int,
        *,
        reward: float = 0.0,
        family_keys: list[str] | tuple[str, ...] = (),
    ) -> None:
        node = self.nodes.setdefault(int(index), LineageNode(index=int(index)))
        node.pulls += 1
        node.reward_total += float(reward)
        node.found_unique_families.update(str(item) for item in family_keys if str(item))

    def rarity_score(self, index: int) -> float:
        node = self.nodes.get(int(index))
        if node is None:
            return 1.0
        siblings = self._sibling_indexes(node)
        sibling_count = len(siblings)
        sibling_pulls = sum(self.nodes[sibling].pulls for sibling in siblings if sibling in self.nodes)
        sibling_reward = sum(
            max(0.0, self.nodes[sibling].reward_total)
            for sibling in siblings
            if sibling in self.nodes
        )
        branch_pulls = self._subtree_pulls(int(index))
        branch_families = self._subtree_family_count(int(index))
        novelty_bonus = min(0.6, 0.15 * branch_families)
        crowding_penalty = 0.18 * sibling_count + 0.08 * sibling_pulls + 0.04 * branch_pulls
        success_crowding_penalty = 0.06 * sibling_reward
        return max(0.0, min(2.0, 1.0 + novelty_bonus - crowding_penalty - success_crowding_penalty))

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "roots": sorted(self.roots),
            "nodes": [node.to_state_dict() for _, node in sorted(self.nodes.items())],
        }

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any] | None) -> "LineageDAG":
        if not isinstance(data, Mapping):
            return cls()
        dag = cls()
        for raw_node in data.get("nodes", []) or []:
            if not isinstance(raw_node, Mapping):
                continue
            node = LineageNode.from_state_dict(raw_node)
            dag.nodes[node.index] = node
        dag.roots = {int(item) for item in data.get("roots", []) or []}
        dag._repair_links()
        return dag

    def _unlink_from_parent(self, index: int, parent_index: int | None) -> None:
        if parent_index is None:
            return
        parent = self.nodes.get(int(parent_index))
        if parent is not None:
            parent.children.discard(int(index))

    def _sibling_indexes(self, node: LineageNode) -> set[int]:
        if node.parent_index is None:
            return {root for root in self.roots if root != node.index}
        parent = self.nodes.get(node.parent_index)
        if parent is None:
            return set()
        return {child for child in parent.children if child != node.index}

    def _subtree_pulls(self, index: int) -> int:
        total = 0
        for node in self._walk_subtree(index):
            total += node.pulls
        return total

    def _subtree_family_count(self, index: int) -> int:
        families: set[str] = set()
        for node in self._walk_subtree(index):
            families.update(node.found_unique_families)
        return len(families)

    def _walk_subtree(self, index: int) -> list[LineageNode]:
        start = self.nodes.get(int(index))
        if start is None:
            return []
        out: list[LineageNode] = []
        queue: deque[int] = deque([int(index)])
        seen: set[int] = set()
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            node = self.nodes.get(current)
            if node is None:
                continue
            out.append(node)
            queue.extend(sorted(node.children))
        return out

    def _repair_links(self) -> None:
        for node in self.nodes.values():
            node.children = {child for child in node.children if child in self.nodes and child != node.index}
        for index, node in list(self.nodes.items()):
            parent = node.parent_index
            if parent is None or parent not in self.nodes or parent == index:
                node.parent_index = None
                self.roots.add(index)
                continue
            self.roots.discard(index)
            self.nodes[parent].children.add(index)
