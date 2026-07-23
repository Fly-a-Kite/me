"""Versioned execution observations and append-only evidence storage.

The legacy run JSONL remains a supported interchange format.  These records
are deliberately separate from it: an :class:`Observation` never overwrites
the raw execution result, and a :class:`Verdict` is a later, versioned view of
one or more observations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from datadiff.experiment_manifest import stable_digest
from datadiff.reproducibility import CaseSeedAssignment, case_digest, program_digest
from datadiff.util import JsonlWriter, dump_json


CASE_MANIFEST_SCHEMA_VERSION = "datadiff-case-manifest-v2"
OBSERVATION_SCHEMA_VERSION = "datadiff-observation-v2"
VERDICT_SCHEMA_VERSION = "datadiff-verdict-v2"
RUN_MANIFEST_SCHEMA_VERSION = "datadiff-run-manifest-v2"


class VerdictStage(str, Enum):
    HARNESS_ERROR = "HarnessError"
    BACKEND_CRASH_TIMEOUT = "BackendCrash/Timeout"
    CAPABILITY_UNSUPPORTED = "CapabilityUnsupported"
    NORMALIZED_COMPARABLE = "NormalizedComparable"
    EXPECTED_SEMANTIC_BOUNDARY = "ExpectedSemanticBoundary"
    CANDIDATE_IMPLEMENTATION_DIVERGENCE = "CandidateImplementationDivergence"
    CONFIRMED_ROOT = "ConfirmedRoot"


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    digest: str
    relative_path: str
    byte_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContentAddressedSidecar:
    """Small JSON content-addressed store for raw results and plans."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def put(self, kind: str, payload: Any) -> ArtifactRef:
        safe_kind = _safe_component(kind)
        body = _canonical_json(payload)
        digest = stable_digest(f"{safe_kind}-artifact", payload)
        relative = Path("sidecars") / safe_kind / f"{digest}.json"
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(path)
        return ArtifactRef(
            artifact_id=digest,
            kind=safe_kind,
            digest=digest,
            relative_path=relative.as_posix(),
            byte_count=len(body.encode("utf-8")),
        )

    def read(self, reference: ArtifactRef | Mapping[str, Any]) -> Any:
        row = reference.to_dict() if isinstance(reference, ArtifactRef) else dict(reference)
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid sidecar path")
        return json.loads((self.root / relative).read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class CaseManifest:
    case_id: str
    seed_assignment: CaseSeedAssignment
    case_digest: str
    program_digest: str
    program_ir_digest: str
    target_ids: tuple[str, ...] = ()
    capability_units: tuple[str, ...] = ()
    schema_version: str = CASE_MANIFEST_SCHEMA_VERSION
    manifest_digest: str = ""

    @classmethod
    def from_case(
        cls,
        case: Mapping[str, Any],
        *,
        root_seed: int,
        case_index: int,
        source: str,
        generator_version: str,
        mutator_version: str = "",
        program_ir_digest: str = "",
        target_ids: Sequence[str] = (),
        capability_units: Sequence[str] = (),
    ) -> "CaseManifest":
        assignment = CaseSeedAssignment.allocate(
            root_seed=root_seed,
            case_index=case_index,
            source=source,
            generator_version=generator_version,
            mutator_version=mutator_version,
        )
        payload = {
            "case_id": str(case.get("case_id", "") or ""),
            "seed_assignment": assignment.to_dict(),
            "case_digest": case_digest(case),
            "program_digest": program_digest(case.get("program", {})),
            "program_ir_digest": str(program_ir_digest or ""),
            "target_ids": sorted({str(value) for value in target_ids if str(value)}),
            "capability_units": sorted({str(value) for value in capability_units if str(value)}),
            "schema_version": CASE_MANIFEST_SCHEMA_VERSION,
        }
        return cls(
            case_id=payload["case_id"],
            seed_assignment=assignment,
            case_digest=payload["case_digest"],
            program_digest=payload["program_digest"],
            program_ir_digest=payload["program_ir_digest"],
            target_ids=tuple(payload["target_ids"]),
            capability_units=tuple(payload["capability_units"]),
            manifest_digest=stable_digest("case-manifest", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seed_assignment"] = self.seed_assignment.to_dict()
        payload["target_ids"] = list(self.target_ids)
        payload["capability_units"] = list(self.capability_units)
        return payload


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    case_manifest_digest: str
    backend: str
    backend_version: str
    target_id: str
    status: str
    raw_result_ref: ArtifactRef | None
    duration_ms: float
    resources: dict[str, float]
    lowering_diagnostics: dict[str, Any]
    schema_version: str = OBSERVATION_SCHEMA_VERSION

    @classmethod
    def from_raw_result(
        cls,
        *,
        case_manifest: CaseManifest,
        backend: str,
        raw_result: Mapping[str, Any],
        backend_version: str = "",
        target_id: str = "",
        sidecars: ContentAddressedSidecar | None = None,
    ) -> "Observation":
        raw = dict(raw_result)
        raw_ref = sidecars.put("raw-result", raw) if sidecars is not None else None
        error_type = str(raw.get("error_type", "") or "")
        error = str(raw.get("error", "") or "")
        diagnostics = {
            "error_type": error_type,
            "is_harness_lowering_error": (
                error_type == "HarnessLoweringError" or "HarnessLoweringError" in error
            ),
            "plan_ref": (
                sidecars.put("plan", raw["physical_plan"]).to_dict()
                if sidecars is not None and isinstance(raw.get("physical_plan"), Mapping)
                else None
            ),
        }
        resources = {
            key: _nonnegative_float(raw.get(key, 0.0))
            for key in ("cpu_ms", "rss_mib", "io_read_bytes", "io_write_bytes")
        }
        identity = {
            "case_manifest_digest": case_manifest.manifest_digest,
            "backend": str(backend),
            "backend_version": str(backend_version),
            "target_id": str(target_id),
            "raw_result_digest": raw_ref.digest if raw_ref else stable_digest("raw-result", raw),
        }
        return cls(
            observation_id=stable_digest("observation", identity),
            case_manifest_digest=case_manifest.manifest_digest,
            backend=str(backend),
            backend_version=str(backend_version),
            target_id=str(target_id),
            status=str(raw.get("status", "unknown") or "unknown"),
            raw_result_ref=raw_ref,
            duration_ms=_nonnegative_float(raw.get("duration_ms", 0.0)),
            resources=resources,
            lowering_diagnostics=diagnostics,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_result_ref"] = (
            self.raw_result_ref.to_dict() if self.raw_result_ref is not None else None
        )
        return payload


@dataclass(frozen=True, slots=True)
class Verdict:
    verdict_id: str
    case_manifest_digest: str
    stage: VerdictStage
    reason_code: str
    evidence_ids: tuple[str, ...]
    oracle_version: str
    observation_ids: tuple[str, ...]
    schema_version: str = VERDICT_SCHEMA_VERSION

    @classmethod
    def decide(
        cls,
        *,
        case_manifest: CaseManifest,
        observations: Sequence[Observation],
        reason_code: str,
        oracle_version: str,
        capability_unsupported: bool = False,
        expected_boundary: bool = False,
        candidate_divergence: bool = False,
        confirmed_root: bool = False,
        evidence_ids: Sequence[str] = (),
    ) -> "Verdict":
        stage = _select_stage(
            observations,
            capability_unsupported=capability_unsupported,
            expected_boundary=expected_boundary,
            candidate_divergence=candidate_divergence,
            confirmed_root=confirmed_root,
        )
        observation_ids = tuple(sorted(observation.observation_id for observation in observations))
        evidence_ids = tuple(sorted({str(value) for value in evidence_ids if str(value)}))
        identity = {
            "case_manifest_digest": case_manifest.manifest_digest,
            "stage": stage.value,
            "reason_code": str(reason_code),
            "evidence_ids": list(evidence_ids),
            "oracle_version": str(oracle_version),
            "observation_ids": list(observation_ids),
        }
        return cls(
            verdict_id=stable_digest("verdict", identity),
            case_manifest_digest=case_manifest.manifest_digest,
            stage=stage,
            reason_code=str(reason_code),
            evidence_ids=evidence_ids,
            oracle_version=str(oracle_version),
            observation_ids=observation_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        payload["evidence_ids"] = list(self.evidence_ids)
        payload["observation_ids"] = list(self.observation_ids)
        return payload


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    method_arm: str
    environment: dict[str, Any]
    targets: tuple[dict[str, Any], ...]
    config: dict[str, Any]
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION
    manifest_digest: str = ""

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        method_arm: str,
        environment: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
    ) -> "RunManifest":
        payload = {
            "run_id": str(run_id),
            "method_arm": str(method_arm),
            "environment": dict(environment),
            "targets": [dict(target) for target in targets],
            "config": dict(config),
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        }
        return cls(
            run_id=payload["run_id"],
            method_arm=payload["method_arm"],
            environment=payload["environment"],
            targets=tuple(payload["targets"]),
            config=payload["config"],
            manifest_digest=stable_digest("run-manifest", payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["targets"] = [dict(target) for target in self.targets]
        return payload


class EvidenceJournal:
    """Canonical append-only event journal with content-addressed sidecars."""

    def __init__(
        self,
        root: str | Path,
        run_manifest: RunManifest,
        *,
        buffered: bool = False,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_manifest = run_manifest
        self.sidecars = ContentAddressedSidecar(self.root)
        self.events_path = self.root / "events.jsonl"
        self.buffered = bool(buffered)
        self._event_writer_context = JsonlWriter(self.events_path, buffer_lines=32)
        self._event_writer = self._event_writer_context.__enter__()
        dump_json(run_manifest.to_dict(), self.root / "run_manifest.json")

    def append(
        self,
        *,
        case_manifest: CaseManifest,
        observations: Sequence[Observation],
        verdicts: Sequence[Verdict],
        causal_signatures: Sequence[Any] = (),
    ) -> None:
        records = [self._event("case_manifest", case_manifest.to_dict())]
        records.extend(
            self._event("observation", observation.to_dict())
            for observation in observations
        )
        records.extend(
            self._event("verdict", verdict.to_dict())
            for verdict in verdicts
        )
        records.extend(
            self._event("causal_signature", signature.to_dict())
            for signature in causal_signatures
            if hasattr(signature, "to_dict")
        )
        for record in records:
            self._event_writer.write(record)
        if not self.buffered:
            self._event_writer.flush()

    def flush(self) -> None:
        self._event_writer.flush()

    def close(self) -> None:
        self._event_writer_context.__exit__(None, None, None)

    def _event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "event_type": event_type,
            "run_manifest_digest": self.run_manifest.manifest_digest,
            "payload": payload,
        }


def _select_stage(
    observations: Sequence[Observation], **flags: bool) -> VerdictStage:
    if any(observation.lowering_diagnostics.get("is_harness_lowering_error") for observation in observations):
        return VerdictStage.HARNESS_ERROR
    statuses = {observation.status.lower() for observation in observations}
    if statuses & {"crash", "timeout"}:
        return VerdictStage.BACKEND_CRASH_TIMEOUT
    if flags["capability_unsupported"]:
        return VerdictStage.CAPABILITY_UNSUPPORTED
    if flags["confirmed_root"]:
        return VerdictStage.CONFIRMED_ROOT
    if flags["candidate_divergence"]:
        return VerdictStage.CANDIDATE_IMPLEMENTATION_DIVERGENCE
    if flags["expected_boundary"]:
        return VerdictStage.EXPECTED_SEMANTIC_BOUNDARY
    return VerdictStage.NORMALIZED_COMPARABLE


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=repr)


def _safe_component(value: str) -> str:
    text = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value))
    return text.strip("-") or "artifact"


def _nonnegative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0
