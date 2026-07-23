from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.experiment_manifest import stable_digest
from datadiff.evidence.causal import CausalSignature


EVIDENCE_SCHEMA_VERSION = "datadiff-evidence-v1"
ROOT_OPPORTUNITY_SCHEMA_VERSION = "root-opportunity-frame-v1"


def build_evidence_envelope(
    *,
    case: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    candidate_recheck: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    method_arm: Mapping[str, Any],
    discovery_signature: str,
) -> dict[str, Any]:
    """Build one lossless evidence envelope without promoting candidates to roots."""

    case_id = str(case.get("case_id", "") or "")
    case_digest = stable_digest("case-evidence", case)
    target_by_backend = {
        str(target.get("backend", target.get("name", "")) or ""): target
        for target in targets
        if isinstance(target, Mapping)
    }
    reproduced = {
        str(value)
        for value in candidate_recheck.get("reproduced_keys", ()) or ()
    }
    non_reproduced = {
        str(value)
        for value in candidate_recheck.get("non_reproduced_keys", ()) or ()
    }

    finding_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    family_ids: set[str] = set()
    for index, finding in enumerate(findings):
        if not isinstance(finding, Mapping):
            continue
        suspicious_backends = tuple(
            str(value)
            for value in finding.get("suspicious_backends", ()) or ()
            if str(value)
        )
        finding_identity = {
            "case_digest": case_digest,
            "kind": str(finding.get("kind", "") or ""),
            "oracle": str(finding.get("oracle", "") or ""),
            "root_cause_label": str(finding.get("root_cause", "unknown") or "unknown"),
            "suspicious_backends": list(suspicious_backends),
            "mismatch_class": str(finding.get("mismatch_class", "") or ""),
        }
        finding_id = stable_digest("finding", finding_identity)
        raw_signature = stable_digest(
            "raw-signature",
            {"discovery_signature": discovery_signature, **finding_identity},
        )
        countable = _is_candidate_finding(finding)
        finding_rows.append(
            {
                "finding_id": finding_id,
                "finding_index": index,
                "raw_signature": raw_signature,
                "countable_candidate": countable,
                **finding_identity,
            }
        )
        if not countable:
            continue

        causal_signature = _causal_signature(
            case=case,
            finding=finding,
            suspicious_backends=suspicious_backends,
            target_by_backend=target_by_backend,
            environment=environment,
            candidate_recheck=candidate_recheck,
        )
        # New-format candidate de-duplication is deliberately keyed only by
        # the causal signature.  The legacy root-label family remains in old
        # JSONL during the double-write migration for compatibility auditing.
        family_id = causal_signature.group_key
        family_ids.add(family_id)
        candidate_id = stable_digest(
            "candidate",
            {"finding_id": finding_id, "raw_signature": raw_signature},
        )
        recheck_key = _finding_recheck_label(finding)
        if recheck_key in non_reproduced:
            reproducibility = "not_reproduced"
        elif recheck_key in reproduced:
            reproducibility = "reproduced"
        else:
            reproducibility = "pending"
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "finding_id": finding_id,
                "candidate_family_id": family_id,
                "causal_signature": causal_signature.to_dict(),
                "raw_signature": raw_signature,
                "reproducibility": reproducibility,
                "root_id": None,
                "real_root_status": "unadjudicated",
                "upstream_confirmation": "pending",
                "root_opportunity": _root_opportunity_frame(
                    case=case,
                    finding=finding,
                    candidate_id=candidate_id,
                    suspicious_backends=suspicious_backends,
                    target_by_backend=target_by_backend,
                    environment=environment,
                ),
            }
        )

    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "case": {
            "case_id": case_id,
            "case_digest": case_digest,
            "frontier": _case_frontier(case),
        },
        "method_arm_id": str(method_arm.get("arm_id", "") or ""),
        "findings": finding_rows,
        "candidates": candidate_rows,
        "candidate_families": [
            {"candidate_family_id": family_id}
            for family_id in sorted(family_ids)
        ],
        # Root creation is an explicit adjudication action. It is never inferred
        # from a finding label, a duplicate signature, or a successful recheck.
        "real_roots": [],
        "independently_confirmed_roots": [],
        "summary": {
            "finding_count": len(finding_rows),
            "candidate_count": len(candidate_rows),
            "candidate_family_count": len(family_ids),
            "real_root_count": 0,
            "independently_confirmed_root_count": 0,
        },
    }
    payload["evidence_digest"] = stable_digest("evidence", payload)
    return payload


def _root_opportunity_frame(
    *,
    case: Mapping[str, Any],
    finding: Mapping[str, Any],
    candidate_id: str,
    suspicious_backends: tuple[str, ...],
    target_by_backend: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    backend_rows = []
    for backend in suspicious_backends:
        target = target_by_backend.get(backend, {})
        backend_rows.append(
            {
                "backend": backend,
                "version": str(
                    environment.get(backend, environment.get(backend.split("_", 1)[0], ""))
                    or ""
                ),
                "family": str(target.get("family", "") or ""),
                "mode": str(target.get("execution_model", "") or ""),
            }
        )
    frame = {
        "schema_version": ROOT_OPPORTUNITY_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "case_id": str(case.get("case_id", "") or ""),
        "frontier": _case_frontier(case),
        "backends": backend_rows,
        "operation": _operation_label(case, finding),
        "contract": {
            "mismatch_class": str(finding.get("mismatch_class", "") or ""),
            "root_cause_label": str(finding.get("root_cause", "unknown") or "unknown"),
        },
        "selection_provenance": _selection_provenance(case),
        "raw_signature_to_root": {
            "root_id": None,
            "status": "unadjudicated",
        },
    }
    frame["frame_digest"] = stable_digest("root-opportunity", frame)
    return frame


def _is_candidate_finding(finding: Mapping[str, Any]) -> bool:
    classification = finding.get("classification", {})
    if isinstance(classification, Mapping):
        if classification.get("counts_as_real_bug") is False:
            return False
        if classification.get("candidate_eligible") is False:
            return False
    return not bool(finding.get("non_reproducible", False))


def _finding_recheck_label(finding: Mapping[str, Any]) -> str:
    suspicious = ",".join(
        sorted(str(value) for value in finding.get("suspicious_backends", ()) or ())
    )
    return "|".join(
        (
            str(finding.get("kind", "") or ""),
            str(finding.get("root_cause", "") or ""),
            suspicious,
            str(finding.get("oracle", "") or ""),
        )
    )


def _case_frontier(case: Mapping[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {"frontier_id": "", "case_index": None}
    return {
        "frontier_id": str(
            metadata.get("frontier_id", metadata.get("schedule_id", "")) or ""
        ),
        "case_index": metadata.get("case_index"),
        "seed": case.get("seed"),
    }


def _selection_provenance(case: Mapping[str, Any]) -> dict[str, Any]:
    metadata = case.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    return {
        key: metadata[key]
        for key in (
            "candidate_source",
            "generator_profile",
            "semantic_objective",
            "seed_lineage",
        )
        if key in metadata
    }


def _operation_label(
    case: Mapping[str, Any],
    finding: Mapping[str, Any],
) -> str:
    explicit = str(finding.get("operation", finding.get("operation_kind", "")) or "")
    if explicit:
        return explicit
    program = case.get("program", {})
    if not isinstance(program, Mapping):
        return ""
    operations = program.get("operations", ())
    if not isinstance(operations, Sequence) or not operations:
        return ""
    last = operations[-1]
    if isinstance(last, Mapping):
        return str(last.get("op", last.get("kind", "")) or "")
    return ""


def _causal_signature(
    *,
    case: Mapping[str, Any],
    finding: Mapping[str, Any],
    suspicious_backends: tuple[str, ...],
    target_by_backend: Mapping[str, Mapping[str, Any]],
    environment: Mapping[str, Any],
    candidate_recheck: Mapping[str, Any],
) -> CausalSignature:
    program = case.get("program", {})
    operations = program.get("operations", ()) if isinstance(program, Mapping) else ()
    operation_sequence = [
        str(operation.get("op", operation.get("kind", "")) or "")
        for operation in operations
        if isinstance(operation, Mapping)
    ]
    backend = "|".join(sorted(suspicious_backends)) or "unknown"
    target = target_by_backend.get(suspicious_backends[0], {}) if suspicious_backends else {}
    recheck_key = _finding_recheck_label(finding)
    reproduced = {
        str(value) for value in candidate_recheck.get("reproduced_keys", ()) or ()
    }
    not_reproduced = {
        str(value) for value in candidate_recheck.get("non_reproduced_keys", ()) or ()
    }
    matrix = {
        recheck_key: (
            "reproduced" if recheck_key in reproduced
            else "not_reproduced" if recheck_key in not_reproduced
            else "pending"
        )
    }
    semantic_axes = finding.get("semantic_axes", ()) or ()
    if isinstance(semantic_axes, (str, bytes, bytearray)):
        semantic_axes = [str(semantic_axes)]
    return CausalSignature.build(
        program_ir=program if isinstance(program, Mapping) else {},
        backend=backend,
        backend_version="|".join(
            str(environment.get(name, environment.get(name.split("_", 1)[0], "")) or "")
            for name in sorted(suspicious_backends)
        ),
        dialect=str(target.get("dialect", "") or ""),
        adapter_version=str(target.get("adapter_version", "") or ""),
        lowering_version=str(target.get("lowering_version", "") or ""),
        normalized_difference={
            "kind": finding.get("kind", ""),
            "mismatch_class": finding.get("mismatch_class", ""),
            "signature": finding.get("signature", ""),
        },
        error_family={
            "error_type": finding.get("error_type", ""),
            "error_family": finding.get("error_family", ""),
        },
        operation_sequence=operation_sequence,
        semantic_axes=semantic_axes,
        reproduction_matrix=matrix,
        minimized_case=case,
        prefix_length=finding.get("minimal_prefix_length"),
    )
