"""Canonical evidence records shared by execution, triage, and analysis."""

from datadiff.evidence.schema import build_evidence_envelope
from datadiff.evidence.causal import CausalSignature, causal_signature_from_row
from datadiff.evidence.model import (
    ArtifactRef,
    CaseManifest,
    ContentAddressedSidecar,
    EvidenceJournal,
    Observation,
    RunManifest,
    Verdict,
    VerdictStage,
)

__all__ = [
    "ArtifactRef",
    "CaseManifest",
    "CausalSignature",
    "ContentAddressedSidecar",
    "EvidenceJournal",
    "Observation",
    "RunManifest",
    "Verdict",
    "VerdictStage",
    "build_evidence_envelope",
    "causal_signature_from_row",
]
