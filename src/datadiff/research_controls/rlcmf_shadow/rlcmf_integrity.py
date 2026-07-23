from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from datadiff.env import collect_environment
from datadiff.research_controls.rlcmf_shadow.rlcmf_manifest import ValidatedRLCMFManifest


RLCMF_CONFIG_FIELDS = (
    "fidelity",
    "audit",
    "coverage_debt",
    "estimators",
    "safety",
    "voi",
    "budget",
    "promotion",
)


def canonical_payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def source_tree_sha256(project_root: Path) -> str:
    root = Path(project_root).resolve()
    rows: list[dict[str, str]] = []
    source_roots = (root / "src" / "datadiff", root / "scripts")
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    registered_files = (
        root / "pyproject.toml",
        root / "experiments" / "schemas" / "rlcmf-manifest-v1.schema.json",
        root / "docs" / "rlcmf_methodology.md",
    )
    for registered_path in registered_files:
        if not registered_path.exists():
            continue
        rows.append(
            {
                "path": str(registered_path.relative_to(root)),
                "sha256": hashlib.sha256(
                    registered_path.read_bytes()
                ).hexdigest(),
            }
        )
    return canonical_payload_sha256(rows)


def config_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in RLCMF_CONFIG_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"RLCMF config contract is missing fields: {missing}")
    return {field: payload[field] for field in RLCMF_CONFIG_FIELDS}


def config_contract_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(config_contract(payload))


def voi_model_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    voi = payload.get("voi")
    if not isinstance(voi, Mapping):
        raise ValueError("RLCMF manifest lacks a VOI contract")
    fields = (
        "model",
        "features",
        "weights",
        "minimum_cost_ms",
        "raise_thresholds",
    )
    missing = [field for field in fields if field not in voi]
    if missing:
        raise ValueError(f"RLCMF VOI model is missing fields: {missing}")
    return {field: voi[field] for field in fields}


def voi_model_contract_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_payload_sha256(voi_model_contract(payload))


def verify_runtime_integrity(
    manifest: ValidatedRLCMFManifest,
    *,
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    implementation = manifest.payload["implementation"]
    expected = {
        "source_sha256": str(implementation["source_sha256"]),
        "environment_sha256": str(implementation["environment_sha256"]),
        "config_sha256": str(implementation["config_sha256"]),
        "model_sha256": str(implementation.get("model_sha256", "")),
    }
    observed = {
        "source_sha256": source_tree_sha256(project_root),
        "environment_sha256": canonical_payload_sha256(
            dict(environment or collect_environment())
        ),
        "config_sha256": config_contract_sha256(manifest.payload),
        "model_sha256": voi_model_contract_sha256(manifest.payload),
    }
    checks = {
        field.removesuffix("_sha256"): expected[field] == observed[field]
        for field in expected
    }
    return {
        "schema_version": "rlcmf-runtime-integrity-v1",
        "manifest_sha256": manifest.sha256,
        "expected": expected,
        "observed": observed,
        "checks": checks,
        "all_pass": all(checks.values()),
    }
