from __future__ import annotations

from typing import Any, Iterable, Mapping

from datadiff.finding_outcomes import candidate_issue_family_keys


def row_has_candidate_signal(row: Mapping[str, Any] | None) -> bool:
    """Return the P4.8-B candidate predicate used before full confirmation."""
    return bool(
        row
        and (
            str(row.get("status", "")) == "bug"
            or bool(row.get("findings", []) or [])
        )
    )


def candidate_family_evidence(row: dict[str, Any]) -> dict[str, object]:
    """Return observed and safely confirmed family identities for a run row."""
    if not row_has_candidate_signal(row):
        return {
            "observed_candidate_families": [],
            "confirmed_candidate_families": [],
            "family_confirmation": "not_a_candidate",
        }
    observed = tuple(
        sorted(candidate_issue_family_keys(row.get("findings", []) or []).keys())
    )
    if not observed:
        return {
            "observed_candidate_families": [],
            "confirmed_candidate_families": [],
            "family_confirmation": "candidate_family_unclassified",
        }
    recheck = row.get("candidate_recheck", {}) or {}
    if int(recheck.get("attempts", 0) or 0) > 0:
        return {
            "observed_candidate_families": list(observed),
            "confirmed_candidate_families": list(observed),
            "family_confirmation": "fresh_backend_recheck",
        }
    backend_sampling = row.get("backend_sampling", {}) or {}
    if (
        backend_sampling.get("confirmation_executed") is True
        and backend_sampling.get("confirmation_candidate_signal") is True
    ):
        return {
            "observed_candidate_families": list(observed),
            "confirmed_candidate_families": list(observed),
            "family_confirmation": "sampled_full_suite_confirmation",
        }
    return {
        "observed_candidate_families": list(observed),
        "confirmed_candidate_families": [],
        "family_confirmation": "no_independent_confirmation",
    }


class NoveltyAwareCandidateBurst:
    """Track auditable candidate-triggered budget bursts.

    Family identities are learned only from the confirmed/full execution row
    supplied by the caller.  A candidate without a confirmed family remains a
    conservative event and therefore keeps the safety burst.
    """

    def __init__(
        self,
        *,
        burst_cases: int,
        novel_only: bool = False,
    ) -> None:
        self.burst_cases = max(0, int(burst_cases or 0))
        self.novel_only = bool(novel_only)
        self.remaining = 0
        self._candidate_events = 0
        self._novel_candidate_events = 0
        self._duplicate_candidate_events = 0
        self._conservative_candidate_events = 0
        self._trigger_events = 0
        self._suppressed_duplicate_events = 0
        self._extension_cases = 0
        self._seen_families: set[str] = set()

    def consume_case(self) -> int:
        if self.remaining > 0:
            self.remaining -= 1
        return self.remaining

    def record(
        self,
        *,
        policy_enabled: bool,
        candidate_detected: bool,
        candidate_families: Iterable[str] = (),
    ) -> dict[str, object]:
        families = tuple(
            sorted(
                {
                    str(family).strip()
                    for family in candidate_families
                    if str(family).strip()
                }
            )
        )
        before = self.remaining
        decision: dict[str, object] = {
            "policy_enabled": bool(policy_enabled),
            "candidate_detected": bool(candidate_detected),
            "classification": "no_candidate",
            "candidate_families": list(families),
            "novel_candidate_families": [],
            "candidate_burst_novel_only": self.novel_only,
            "candidate_burst_cases": self.burst_cases,
            "burst_triggered": False,
            "burst_suppressed": False,
            "burst_remaining_before": before,
            "burst_remaining_after": before,
            "burst_extension_cases": 0,
            "reason": "no candidate signal",
        }
        if not policy_enabled:
            decision["classification"] = "policy_disabled"
            decision["reason"] = "adaptive sampling policy is disabled"
            return decision
        if not candidate_detected:
            return decision

        self._candidate_events += 1
        novel_families = tuple(
            family for family in families if family not in self._seen_families
        )
        self._seen_families.update(families)
        decision["novel_candidate_families"] = list(novel_families)

        if not families:
            classification = "conservative_unclassified_candidate"
            self._conservative_candidate_events += 1
            should_trigger = True
            reason = (
                "candidate lacks a confirmed family identity; retain the safety burst"
            )
        elif novel_families:
            classification = "novel_confirmed_family"
            self._novel_candidate_events += 1
            should_trigger = True
            reason = "first-seen confirmed family renews the full-budget burst"
        else:
            classification = "duplicate_confirmed_family"
            self._duplicate_candidate_events += 1
            should_trigger = not self.novel_only
            reason = (
                "repeated confirmed family does not renew a novelty-only burst"
                if self.novel_only
                else "default compatibility policy renews on every candidate event"
            )

        decision["classification"] = classification
        if not should_trigger:
            self._suppressed_duplicate_events += 1
            decision["burst_suppressed"] = True
            decision["reason"] = reason
            return decision
        if self.burst_cases <= 0:
            decision["reason"] = f"{reason}; configured burst length is zero"
            return decision

        self.remaining = max(self.remaining, self.burst_cases)
        extension = self.remaining - before
        self._trigger_events += 1
        self._extension_cases += extension
        decision.update(
            {
                "burst_triggered": True,
                "burst_remaining_after": self.remaining,
                "burst_extension_cases": extension,
                "reason": reason,
            }
        )
        return decision

    def summary(self) -> dict[str, object]:
        return {
            "candidate_events": self._candidate_events,
            "novel_candidate_events": self._novel_candidate_events,
            "duplicate_candidate_events": self._duplicate_candidate_events,
            "conservative_candidate_events": self._conservative_candidate_events,
            "candidate_burst_trigger_events": self._trigger_events,
            "candidate_burst_suppressed_duplicate_events": (
                self._suppressed_duplicate_events
            ),
            "candidate_burst_extension_cases": self._extension_cases,
            "candidate_burst_novel_only": self.novel_only,
            "seen_candidate_families": sorted(self._seen_families),
            "candidate_burst_remaining": self.remaining,
        }
