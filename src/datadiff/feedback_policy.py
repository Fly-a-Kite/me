from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff.case_features import PROBE_ROOTS
from datadiff.dsl import Case
from datadiff.operation_semantics import op_kind

CALIBRATION_PROBE_OPS = frozenset(PROBE_ROOTS) | {
    "running_sum",
    "tuple_absence_filter",
}


@dataclass(frozen=True, slots=True)
class FeedbackStoragePolicy:
    allow_feedback_mutation_child_candidate_bug: bool = False
    max_feedback_mutation_child_depth: int = 2


@dataclass(frozen=True, slots=True)
class FeedbackStorageContext:
    candidate_source: str = "generated"
    seed_lineage: dict[str, Any] | None = None
    candidate_bug_families: tuple[str, ...] = ()

    @property
    def lineage_depth(self) -> int:
        if not isinstance(self.seed_lineage, dict):
            return 0
        try:
            return max(0, int(self.seed_lineage.get("depth", 0) or 0))
        except (TypeError, ValueError):
            return 0

    @property
    def has_candidate_bug_family(self) -> bool:
        return any(str(family).strip() for family in self.candidate_bug_families)

    @property
    def is_feedback_child(self) -> bool:
        return self.candidate_source == "feedback_mutation" or self.lineage_depth > 0


def feedback_storage_decision(
    case: Case,
    context: FeedbackStorageContext,
    policy: FeedbackStoragePolicy | None = None,
) -> tuple[bool, str]:
    policy = policy or FeedbackStoragePolicy()
    if _is_calibration_probe_case(case):
        return False, "calibration_probe_case"
    if not context.is_feedback_child:
        return True, ""
    if (
        policy.allow_feedback_mutation_child_candidate_bug
        and context.has_candidate_bug_family
        and context.lineage_depth <= max(0, int(policy.max_feedback_mutation_child_depth or 0))
    ):
        return True, ""
    return False, "feedback_mutation_child"


def source_scheduler_snapshot(feedback: Any | None) -> list[dict[str, Any]]:
    scheduler = getattr(feedback, "source_scheduler", None) if feedback is not None else None
    if scheduler is None or not hasattr(scheduler, "snapshot"):
        return []
    return scheduler.snapshot()


def _is_calibration_probe_case(case: Case) -> bool:
    return any(op_kind(op) in CALIBRATION_PROBE_OPS for op in case.program.operations)
