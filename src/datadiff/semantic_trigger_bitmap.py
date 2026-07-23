"""Exact compact first-seen index for registered semantic witness trigger cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from datadiff.boundary_values import boundary_profiles
from datadiff.compact_novelty import DenseBitmap
from datadiff.goal_first import generation_goals
from datadiff.goal_first_variants import (
    SEMANTIC_WITNESS_DATA_PATTERN_COUNT,
    all_witness_builder_variants,
)


SEMANTIC_TRIGGER_BITMAP_SCHEMA_VERSION = "semantic-trigger-bitmap-v3"
_ACTIVATION_STATUSES = ("activated", "not_activated", "not_evaluated")


@dataclass(frozen=True, slots=True)
class TriggerCellObservation:
    registered: bool
    cell_index: int | None
    first_seen: bool | None
    goal_id: str
    variant_id: str
    data_pattern_index: str
    boundary_profile: str
    activation_status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEMANTIC_TRIGGER_BITMAP_SCHEMA_VERSION,
            "registered": self.registered,
            "cell_index": self.cell_index,
            "first_seen": self.first_seen,
            "goal_id": self.goal_id,
            "variant_id": self.variant_id,
            "data_pattern_index": self.data_pattern_index,
            "boundary_profile": self.boundary_profile,
            "activation_status": self.activation_status,
            "reason": self.reason,
        }


class SemanticTriggerBitmap:
    """Shadow index only; it never discards a case or adjudicates a bug."""

    __slots__ = (
        "_activation_lookup",
        "_bitmap",
        "_boundary_lookup",
        "_goal_variant_pattern_lookup",
        "_status_count",
        "_boundary_count",
        "unregistered_observations",
    )

    def __init__(self) -> None:
        goals = generation_goals()
        goal_variant_pattern_cells = tuple(
            (goal.goal_id, variant_id, data_pattern_index)
            for goal in goals
            for variant_id, data_pattern_indexes in (
                ("legacy_single_builder", ("<legacy>",)),
                *(
                    (
                        variant.variant_id,
                        tuple(
                            str(index)
                            for index in range(SEMANTIC_WITNESS_DATA_PATTERN_COUNT)
                        ),
                    )
                    for variant in all_witness_builder_variants(goal.goal_id)
                ),
            )
            for data_pattern_index in data_pattern_indexes
        )
        boundary_ids = (
            "<none>",
            *tuple(profile.profile_id for profile in boundary_profiles()),
        )
        self._goal_variant_pattern_lookup = {
            value: index for index, value in enumerate(goal_variant_pattern_cells)
        }
        self._boundary_lookup = {
            value: index for index, value in enumerate(boundary_ids)
        }
        self._activation_lookup = {
            value: index for index, value in enumerate(_ACTIVATION_STATUSES)
        }
        self._boundary_count = len(boundary_ids)
        self._status_count = len(_ACTIVATION_STATUSES)
        self._bitmap = DenseBitmap(
            len(goal_variant_pattern_cells)
            * self._boundary_count
            * self._status_count
        )
        self.unregistered_observations = 0

    def observe_trace(self, trace: Mapping[str, Any]) -> TriggerCellObservation:
        goal_id = str(
            (trace.get("selected_goal", {}) or {}).get("goal_id", "") or ""
        )
        variant_id = str(
            (trace.get("builder_variant", {}) or {}).get("variant_id", "") or ""
        )
        data_pattern_index = str(
            (
                (trace.get("builder_variant", {}) or {}).get("data_pattern", {})
                or {}
            ).get("pattern_index", "<legacy>")
        )
        boundary_profile = str(
            (trace.get("boundary_application", {}) or {}).get("profile_id", "")
            or "<none>"
        )
        activation_status = str(
            (trace.get("semantic_activation", {}) or {}).get(
                "evaluation_status", "not_evaluated"
            )
            or "not_evaluated"
        )
        try:
            goal_variant_pattern_index = self._goal_variant_pattern_lookup[
                (goal_id, variant_id, data_pattern_index)
            ]
            boundary_index = self._boundary_lookup[boundary_profile]
            activation_index = self._activation_lookup[activation_status]
        except KeyError as exc:
            self.unregistered_observations += 1
            return TriggerCellObservation(
                registered=False,
                cell_index=None,
                first_seen=None,
                goal_id=goal_id,
                variant_id=variant_id,
                data_pattern_index=data_pattern_index,
                boundary_profile=boundary_profile,
                activation_status=activation_status,
                reason=f"unregistered_trigger_dimension:{exc}",
            )
        cell_index = (
            (goal_variant_pattern_index * self._boundary_count + boundary_index)
            * self._status_count
            + activation_index
        )
        first_seen = self._bitmap.observe(cell_index)
        return TriggerCellObservation(
            registered=True,
            cell_index=cell_index,
            first_seen=first_seen,
            goal_id=goal_id,
            variant_id=variant_id,
            data_pattern_index=data_pattern_index,
            boundary_profile=boundary_profile,
            activation_status=activation_status,
        )

    def snapshot(self, *, include_data: bool = False) -> dict[str, Any]:
        return {
            **self._bitmap.snapshot(include_data=include_data),
            "schema_version": SEMANTIC_TRIGGER_BITMAP_SCHEMA_VERSION,
            "dimension_names": [
                "registered_goal_variant_pattern_cell",
                "boundary_profile",
                "activation_status",
            ],
            "dimension_sizes": [
                len(self._goal_variant_pattern_lookup),
                self._boundary_count,
                self._status_count,
            ],
            "dimension_count": 3,
            "registered_goal_variant_pattern_cell_count": len(
                self._goal_variant_pattern_lookup
            ),
            "unregistered_observation_count": self.unregistered_observations,
            "behavioral_role": "shadow_first_seen_hint_only",
            "case_discard_authority": False,
            "confirmed_bug_dedupe_authority": False,
        }
