"""Bridge legacy run rows to compact, append-only event evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datadiff.evidence import (
    CaseManifest,
    EvidenceJournal,
    Observation,
    RunManifest,
    Verdict,
    causal_signature_from_row,
)
from datadiff.util import jsonl_log_stem


def event_evidence_dir(run_file: Path) -> Path:
    return run_file.with_name(f"{jsonl_log_stem(run_file)}.evidence")


def create_event_journal(
    *,
    run_file: Path,
    run_id: str,
    method_arm: Mapping[str, Any],
    environment: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> EvidenceJournal:
    manifest = RunManifest.create(
        run_id=run_id,
        method_arm=str(method_arm.get("arm_id", "") or ""),
        environment=environment,
        targets=targets,
        config=config,
    )
    return EvidenceJournal(event_evidence_dir(run_file), manifest, buffered=True)


def append_case_events(
    journal: EvidenceJournal,
    *,
    row: Mapping[str, Any],
    root_seed: int,
    case_index: int,
    selected_meta: Mapping[str, Any],
    capability_units: Sequence[str],
) -> dict[str, Any]:
    case = _mapping(row.get("case"))
    targets = [item for item in row.get("targets", ()) if isinstance(item, Mapping)]
    source = _canonical_source(selected_meta.get("source"))
    program_ir = _mapping(row.get("program_ir"))
    manifest = CaseManifest.from_case(
        case,
        root_seed=root_seed,
        case_index=case_index,
        source=source,
        generator_version=str(row.get("method_arm", {}).get("method_version", "") or ""),
        mutator_version=str(selected_meta.get("mutation", {}).get("version", "") or ""),
        program_ir_digest=str(program_ir.get("digest", "") or ""),
        target_ids=[str(target.get("backend", target.get("name", "")) or "") for target in targets],
        capability_units=capability_units,
    )
    observations = _observations(journal, manifest, row, targets)
    verdicts, signatures = _verdicts_and_signatures(manifest, observations, row)
    journal.append(
        case_manifest=manifest,
        observations=observations,
        verdicts=verdicts,
        causal_signatures=signatures,
    )
    return {
        "case_manifest_digest": manifest.manifest_digest,
        "observation_ids": [observation.observation_id for observation in observations],
        "verdict_ids": [verdict.verdict_id for verdict in verdicts],
        "causal_signatures": [signature.group_key for signature in signatures],
    }


def _observations(
    journal: EvidenceJournal,
    manifest: CaseManifest,
    row: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
) -> list[Observation]:
    target_by_backend = {
        str(target.get("backend", target.get("name", "")) or ""): target
        for target in targets
    }
    environment = _mapping(row.get("environment"))
    raw_results = _mapping(row.get("raw_results"))
    observations: list[Observation] = []
    for backend, raw in sorted(raw_results.items()):
        if not isinstance(raw, Mapping):
            continue
        name = str(backend)
        target = target_by_backend.get(name, {})
        version = str(environment.get(name, environment.get(name.split("_", 1)[0], "")) or "")
        observations.append(
            Observation.from_raw_result(
                case_manifest=manifest,
                backend=name,
                backend_version=version,
                target_id=str(target.get("target_id", name) or name),
                raw_result=raw,
                sidecars=journal.sidecars,
            )
        )
    return observations


def _verdicts_and_signatures(
    manifest: CaseManifest,
    observations: Sequence[Observation],
    row: Mapping[str, Any],
) -> tuple[list[Verdict], list[Any]]:
    findings = [item for item in row.get("findings", ()) if isinstance(item, Mapping)]
    if not findings:
        return [
            Verdict.decide(
                case_manifest=manifest,
                observations=observations,
                reason_code="no_oracle_finding",
                oracle_version=_oracle_version(row),
            )
        ], []
    verdicts: list[Verdict] = []
    signatures: list[Any] = []
    for index, finding in enumerate(findings):
        triage = str(finding.get("triage_verdict", "") or "")
        candidate = triage == "candidate_implementation_bug"
        expected = triage in {"documented_semantic_divergence", "expected_semantic_divergence"}
        suspicious = sorted(str(item) for item in finding.get("suspicious_backends", ()) if str(item))
        finding_signatures = [
            causal_signature_from_row(row, finding, backend=backend)
            for backend in suspicious
        ]
        signatures.extend(finding_signatures)
        verdicts.append(
            Verdict.decide(
                case_manifest=manifest,
                observations=observations,
                reason_code=triage or str(finding.get("kind", "oracle_finding") or "oracle_finding"),
                oracle_version=_oracle_version(row),
                capability_unsupported=triage == "capability_unsupported",
                expected_boundary=expected,
                candidate_divergence=candidate,
                confirmed_root=triage == "confirmed_root",
                evidence_ids=[
                    f"finding-{index}",
                    *(signature.group_key for signature in finding_signatures),
                ],
            )
        )
    return verdicts, signatures


def _oracle_version(row: Mapping[str, Any]) -> str:
    config = _mapping(row.get("config"))
    return str(config.get("oracle_version", "datadiff-oracle-v2") or "datadiff-oracle-v2")


def _canonical_source(value: Any) -> str:
    source = str(value or "").lower()
    if "metamorphic" in source or "semantic" in source:
        return "semantic_metamorphic_mutation"
    if "replay" in source or "regression" in source:
        return "known_regression_replay"
    if "feedback" in source or "mutation" in source:
        return "feedback_mutation"
    return "fresh_grammar_generation"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
