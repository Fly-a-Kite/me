from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datadiff_osc._canonical import stable_digest, to_primitive


@dataclass(frozen=True, slots=True)
class HardeningProposal:
    proposal_id: str
    source_counterexample_digest: str
    affected_component: str
    proposed_relation: str
    proposed_preconditions: tuple[str, ...]
    evidence_sources: tuple[str, ...]
    target_schema_version: str
    status: str = "pending"
    rejection_reason: str = ""
    schema_version: str = "osc-offline-hardening-proposal-v1"

    def __post_init__(self) -> None:
        if self.status not in {"pending", "accepted", "rejected"}:
            raise ValueError(f"invalid hardening proposal status: {self.status}")
        if self.status == "accepted" and not self.evidence_sources:
            raise ValueError("accepted hardening proposal needs independent evidence")
        if self.status == "rejected" and not self.rejection_reason:
            raise ValueError("rejected hardening proposal needs a reason")

    @property
    def digest(self) -> str:
        return stable_digest("osc-hardening-proposal", self)

    def to_dict(self) -> dict[str, Any]:
        return to_primitive(self)


def assert_not_live_mutation(*, campaign_frozen: bool) -> None:
    if campaign_frozen:
        raise RuntimeError(
            "offline counterexample-guided hardening cannot mutate a frozen campaign"
        )

