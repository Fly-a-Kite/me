from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datadiff.canonicalization import short_canonical_hash


EXPERIMENT_MANIFEST_SCHEMA_VERSION = "latticefuzz-experiment-manifest-v1"
CONFIG_MANIFEST_SCHEMA_VERSION = "latticefuzz-config-manifest-v1"


def stable_digest(prefix: str, payload: Any, *, length: int = 64) -> str:
    return f"{prefix}-{short_canonical_hash(payload, length)}"


def finalize_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("config_digest", None)
    out["config_schema_version"] = CONFIG_MANIFEST_SCHEMA_VERSION
    out["config_digest"] = stable_digest("config", out)
    return out


def build_case_experiment_manifest(
    *,
    case_payload: Mapping[str, Any],
    backends: Sequence[str],
    target_specs: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    config_payload: Mapping[str, Any],
    ir_mode: str = "",
    program_ir_digest: str = "",
) -> dict[str, Any]:
    method_arm_id, method_arm_digest = _method_identity(config_payload)
    case_digest = stable_digest("case", case_payload)
    backend_digest = stable_digest("backends", [str(backend) for backend in backends])
    target_digest = stable_digest("targets", list(target_specs))
    environment_digest = stable_digest("environment", dict(environment))
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "case_id": str(case_payload.get("case_id", "") or ""),
        "case_digest": case_digest,
        "backends": [str(backend) for backend in backends],
        "backend_digest": backend_digest,
        "target_digest": target_digest,
        "environment_digest": environment_digest,
        "config_digest": str(config_payload.get("config_digest", "") or ""),
        "method_arm_id": method_arm_id,
        "method_arm_digest": method_arm_digest,
        "ir_mode": str(ir_mode or ""),
        "program_ir_digest": str(program_ir_digest or ""),
    }
    payload["comparison_block_digest"] = stable_digest(
        "comparison-block",
        {
            "case_digest": case_digest,
            "backend_digest": backend_digest,
            "target_digest": target_digest,
            "environment_digest": environment_digest,
        },
    )
    payload["manifest_digest"] = stable_digest("experiment", payload)
    return payload


def build_run_experiment_manifest(
    *,
    seed: int,
    cases: int | None,
    duration_s: float | None,
    backends: Sequence[str],
    target_specs: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
    config_payload: Mapping[str, Any],
) -> dict[str, Any]:
    method_arm_id, method_arm_digest = _method_identity(config_payload)
    backend_digest = stable_digest("backends", [str(backend) for backend in backends])
    target_digest = stable_digest("targets", list(target_specs))
    environment_digest = stable_digest("environment", dict(environment))
    provenance_digest = stable_digest("provenance", dict(run_provenance))
    payload = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "seed": int(seed),
        "case_budget": None if cases is None else int(cases),
        "duration_budget_s": None if duration_s is None else float(duration_s),
        "backends": [str(backend) for backend in backends],
        "backend_digest": backend_digest,
        "target_digest": target_digest,
        "environment_digest": environment_digest,
        "provenance_digest": provenance_digest,
        "config_digest": str(config_payload.get("config_digest", "") or ""),
        "method_arm_id": method_arm_id,
        "method_arm_digest": method_arm_digest,
    }
    payload["comparison_block_digest"] = stable_digest(
        "comparison-block",
        {
            "seed": int(seed),
            "case_budget": None if cases is None else int(cases),
            "duration_budget_s": None if duration_s is None else float(duration_s),
            "backend_digest": backend_digest,
            "target_digest": target_digest,
            "environment_digest": environment_digest,
            "provenance_digest": provenance_digest,
        },
    )
    payload["manifest_digest"] = stable_digest("experiment", payload)
    return payload


def _method_identity(config_payload: Mapping[str, Any]) -> tuple[str, str]:
    manifest = config_payload.get("method_arm_manifest", {})
    if not isinstance(manifest, Mapping):
        return str(config_payload.get("method_arm", "") or ""), ""
    return (
        str(manifest.get("arm_id", config_payload.get("method_arm", "")) or ""),
        str(manifest.get("digest", "") or ""),
    )
