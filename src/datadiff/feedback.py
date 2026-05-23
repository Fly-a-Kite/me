from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from datadiff.dsl import Case
from datadiff.mutator import mutate_case_with_metadata
from datadiff.scheduler import LocalSourceScheduler
from datadiff.util import CORPUS_DIR, dump_json


@dataclass(slots=True)
class FeedbackState:
    max_corpus: int = 256
    persist_to_disk: bool = False
    max_persisted: int = 4096
    max_cases_per_candidate_family: int = 8
    seen_signatures: set[str] = field(default_factory=set)
    interesting_cases: list[Case] = field(default_factory=list)
    stored_candidate_bug_families: Counter[str] = field(default_factory=Counter)
    persisted_count: int = 0
    last_persisted_to_disk: bool = False
    last_record_skip_reason: str = ""
    source_scheduler: LocalSourceScheduler | None = None
    last_candidate_source: str = "generated"
    last_candidate_metadata: dict[str, Any] = field(default_factory=dict)
    last_source_reward: float | None = None

    def choose_case(self, seed: int, generated: Case) -> Case:
        self.last_candidate_source = "generated"
        self.last_candidate_metadata = _generated_candidate_metadata(generated)
        if not self.interesting_cases:
            return generated
        if self.source_scheduler is None:
            if seed % 3 != 0:
                return generated
        else:
            source = self.source_scheduler.choose_source(feedback_available=bool(self.interesting_cases))
            self.last_candidate_source = source
            if source != "feedback_mutation":
                return generated
        for attempt in range(min(4, len(self.interesting_cases))):
            mutation_seed = seed + attempt
            base = self.interesting_cases[mutation_seed % len(self.interesting_cases)]
            result = mutate_case_with_metadata(base, mutation_seed, allow_probe_operators=False)
            if result.metadata.get("mutation", {}).get("changed"):
                self.last_candidate_source = "feedback_mutation"
                self.last_candidate_metadata = result.metadata
                return result.case
        self.last_candidate_source = "generated"
        self.last_candidate_metadata = _generated_candidate_metadata(generated)
        return generated

    def record(
        self,
        case: Case,
        behavior_signature: str,
        has_finding: bool,
        *,
        candidate_bug_families: list[str] | None = None,
    ) -> bool:
        self.last_persisted_to_disk = False
        self.last_record_skip_reason = ""
        is_new = behavior_signature not in self.seen_signatures
        self.seen_signatures.add(behavior_signature)
        if not (is_new or has_finding):
            self.last_record_skip_reason = "duplicate_uninteresting_behavior"
            return False
        family_keys = _unique_nonempty(candidate_bug_families or [])
        family_limit = max(0, int(self.max_cases_per_candidate_family))
        if family_limit and family_keys and all(
            self.stored_candidate_bug_families[family] >= family_limit for family in family_keys
        ):
            self.last_record_skip_reason = "candidate_family_saturated"
            return False
        if len(self.interesting_cases) < self.max_corpus:
            self.interesting_cases.append(case)
        elif has_finding:
            self.interesting_cases[int(behavior_signature, 16) % self.max_corpus] = case
        else:
            self.last_record_skip_reason = "corpus_full"
            return False
        self.stored_candidate_bug_families.update(family_keys)
        if self.persist_to_disk and self.persisted_count < max(0, self.max_persisted):
            self._write_interesting_case(case, behavior_signature, has_finding)
            self.persisted_count += 1
            self.last_persisted_to_disk = True
        return True

    def record_candidate_result(
        self,
        candidate_source: str,
        *,
        has_finding: bool,
        is_new_behavior: bool,
        preflight: dict[str, Any],
        candidate_bug: bool = False,
        semantic_divergence: bool = False,
        false_positive: bool = False,
        candidate_bug_families: list[str] | None = None,
        candidate_bug_signatures: list[str] | None = None,
    ) -> float | None:
        if self.source_scheduler is None:
            self.last_source_reward = None
            return None
        reward = self.source_scheduler.record_result(
            "feedback_mutation" if candidate_source == "feedback_mutation" else "generated",
            has_finding=has_finding,
            is_new_behavior=is_new_behavior,
            preflight_valid=bool(preflight.get("valid", True)),
            fallback_used=bool(preflight.get("fallback_used", False)),
            candidate_bug=candidate_bug,
            semantic_divergence=semantic_divergence,
            false_positive=false_positive,
            candidate_bug_families=candidate_bug_families,
            candidate_bug_signatures=candidate_bug_signatures,
        )
        self.last_source_reward = reward
        return reward

    def _write_interesting_case(self, case: Case, behavior_signature: str, has_finding: bool) -> None:
        path = CORPUS_DIR / "interesting" / f"{behavior_signature}.json"
        dump_json(
            {
                "behavior_signature": behavior_signature,
                "has_finding": has_finding,
                "case": case.to_dict(),
            },
            path,
        )


def _generated_candidate_metadata(case: Case) -> dict[str, Any]:
    return {
        "seed_lineage": {
            "root_seed": case.seed,
            "parent_seed": None,
            "parent_case_id": "",
            "mutation_seed": None,
            "depth": 0,
        },
        "mutation": {
            "operator": "generated",
            "detail": "generated",
            "changed": False,
        },
    }


def _unique_nonempty(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out
