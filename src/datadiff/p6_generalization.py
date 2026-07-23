from __future__ import annotations

from collections.abc import Sequence
import copy
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

from datadiff.adapter_conformance import run_adapter_conformance
from datadiff.backends import make_backend
from datadiff.backends.pyarrow_backend import PyArrowBackend
from datadiff.ccs_ir import case_to_ccs_ir
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.execution import BackendExecutor
from datadiff.execution_cache import ExecutionResultCache
from datadiff.goal_first import generate_goal_first_case
from datadiff.normalizer import NormalizedResult, normalize_result
from datadiff.oracle import evaluate_case
from datadiff.targets import target_spec


P6_CASE_CORPUS_SCHEMA_VERSION = "p6-unseen-case-corpus-v1"
P6_BACKEND_AUDIT_SCHEMA_VERSION = "p6-unseen-backend-audit-v1"
P6_MODE_AUDIT_SCHEMA_VERSION = "p6-unseen-mode-audit-v1"
P6_LAYOUT_AUDIT_SCHEMA_VERSION = "p6-unseen-layout-audit-v1"
P6_SENTINEL_SCHEMA_VERSION = "p6-seeded-sentinel-v1"


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_frozen_case_rows(
    seed_blocks: Sequence[int],
    *,
    cases_per_block: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_index, block_seed in enumerate(seed_blocks):
        for case_offset in range(int(cases_per_block)):
            seed = int(block_seed) + case_offset
            generated = generate_goal_first_case(
                seed,
                profile="p6_unseen_generalization",
                boundary_mode="fault_model_targeted",
            )
            if generated.case is None:
                raise RuntimeError(
                    f"P6 goal-first case is not constructible seed={seed}: "
                    f"{generated.trace.get('skip_reason', '')}"
                )
            case = generated.case
            payload = case.to_dict()
            rows.append(
                {
                    "schema_version": P6_CASE_CORPUS_SCHEMA_VERSION,
                    "block_index": block_index,
                    "block_seed": int(block_seed),
                    "case_offset": case_offset,
                    "seed": seed,
                    "case_digest": canonical_sha256(payload),
                    "goal_id": str(case.metadata.get("goal_id", "")),
                    "case": payload,
                    "generation_trace": generated.trace,
                }
            )
    return rows


def load_cases(rows: Sequence[dict[str, Any]]) -> list[Case]:
    return [Case.from_dict(dict(row["case"])) for row in rows]


def frozen_method_config() -> ExperimentConfig:
    return ExperimentConfig(
        method_arm="p5_promoted_method",
        enable_parallel_backend_execution=False,
        enable_backend_session_reuse=False,
        enable_metamorphic_oracle=False,
        candidate_recheck_count=0,
        enable_artifact=False,
        enable_feedback=False,
        enable_champion_corpus=False,
        enable_lhs_seeding=False,
        log_level="minimal",
    )


def run_backend_full_audit(
    cases: Sequence[Case],
    *,
    backends: Sequence[str] = ("pandas", "duckdb", "sqlite", "chdb"),
) -> dict[str, Any]:
    config = frozen_method_config()
    executor = BackendExecutor(list(backends))
    rows: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for case in cases:
        required = case_to_ccs_ir(case).required_capabilities
        decisions = {
            backend: target_spec(backend)
            .capability_decision(required_tokens=required)
            .to_dict()
            for backend in backends
        }
        raw, normalized = executor.execute(case, config)
        chdb_applicable = bool(decisions["chdb"]["supported"])
        chdb_ok = normalized["chdb"].status == "ok"
        controls_ok = all(normalized[name].status == "ok" for name in backends if name != "chdb")
        pair_findings: dict[str, list[dict[str, Any]]] = {}
        if chdb_ok and controls_ok:
            for control in (name for name in backends if name != "chdb"):
                findings = evaluate_case(
                    case,
                    {control: normalized[control], "chdb": normalized["chdb"]},
                    comparison_mode="contract",
                )
                pair_findings[control] = [finding.to_dict() for finding in findings]
        agreement = bool(chdb_ok and controls_ok and not any(pair_findings.values()))
        rows.append(
            {
                "case_id": case.case_id,
                "case_digest": canonical_sha256(case.to_dict()),
                "case": case.to_dict(),
                "required_capabilities": list(required),
                "capability_decisions": decisions,
                "chdb_applicable": chdb_applicable,
                "chdb_ok": chdb_ok,
                "controls_ok": controls_ok,
                "agreement": agreement,
                "pair_findings": pair_findings,
                "raw": raw,
                "normalized": {
                    backend: result.to_dict()
                    for backend, result in normalized.items()
                },
            }
        )
    return {
        "schema_version": P6_BACKEND_AUDIT_SCHEMA_VERSION,
        "backends": list(backends),
        "case_count": len(rows),
        "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
        "process_cpu_ms": (time.process_time() - cpu_started) * 1000.0,
        "rows": rows,
    }


def run_polars_mode_audit(
    cases: Sequence[Case],
    *,
    package_version: str,
) -> dict[str, Any]:
    config = frozen_method_config()
    executor = BackendExecutor(["polars_lazy", "polars_streaming"])
    rows: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for case in cases:
        raw, normalized = executor.execute(case, config)
        valid = all(result.status == "ok" for result in normalized.values())
        findings = (
            evaluate_case(case, normalized, comparison_mode="contract") if valid else []
        )
        rows.append(
            {
                "case_id": case.case_id,
                "case_digest": canonical_sha256(case.to_dict()),
                "valid": valid,
                "agreement": bool(valid and not findings),
                "findings": [finding.to_dict() for finding in findings],
                "raw": raw,
                "normalized": {
                    backend: result.to_dict()
                    for backend, result in normalized.items()
                },
            }
        )
    return {
        "schema_version": P6_MODE_AUDIT_SCHEMA_VERSION,
        "package": "polars",
        "package_version": str(package_version),
        "modes": ["lazy", "streaming"],
        "case_count": len(rows),
        "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
        "process_cpu_ms": (time.process_time() - cpu_started) * 1000.0,
        "rows": rows,
    }


def run_pyarrow_layout_audit(
    cases: Sequence[Case],
    *,
    package_version: str,
    layouts: Sequence[str] = ("contiguous", "chunked", "sliced", "dictionary"),
) -> dict[str, Any]:
    config = frozen_method_config()
    rows: list[dict[str, Any]] = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for case in cases:
        resolved = case_to_ccs_ir(case)
        normalized: dict[str, NormalizedResult] = {}
        raw: dict[str, dict[str, Any]] = {}
        applicability: dict[str, dict[str, Any]] = {}
        has_string = any(column.type == "str" for table in case.tables for column in table.columns)
        for layout in layouts:
            if layout == "dictionary" and not has_string:
                applicability[layout] = {
                    "applicable": False,
                    "skip_reason": "layout_not_applicable:dictionary_requires_string",
                }
                continue
            decision = target_spec("pyarrow").capability_decision(
                required_tokens=resolved.required_capabilities,
                physical_layout=layout,
            )
            applicability[layout] = {
                "applicable": decision.supported,
                "skip_reason": decision.skip_reason,
            }
            if not decision.supported:
                continue
            backend = PyArrowBackend()
            backend.configure_physical_layout(layout)
            result = backend.run(case.tables, resolved.to_program())
            result.capability_decision = decision.to_dict()
            key = f"pyarrow_{layout}"
            raw[key] = result.summary_dict()
            normalized[key] = normalize_result(
                result,
                resolved.to_program(),
                enable_normalizer=config.enable_normalizer,
            )
        control = normalized.get("pyarrow_contiguous")
        comparisons: dict[str, dict[str, Any]] = {}
        for layout in layouts:
            if layout == "contiguous" or f"pyarrow_{layout}" not in normalized or control is None:
                continue
            treatment = normalized[f"pyarrow_{layout}"]
            valid = control.status == "ok" and treatment.status == "ok"
            findings = (
                evaluate_case(
                    case,
                    {"pyarrow_contiguous": control, f"pyarrow_{layout}": treatment},
                    comparison_mode="contract",
                )
                if valid
                else []
            )
            comparisons[layout] = {
                "valid": valid,
                "agreement": bool(valid and not findings),
                "findings": [finding.to_dict() for finding in findings],
            }
        rows.append(
            {
                "case_id": case.case_id,
                "case_digest": canonical_sha256(case.to_dict()),
                "applicability": applicability,
                "comparisons": comparisons,
                "raw": raw,
                "normalized": {
                    key: result.to_dict() for key, result in normalized.items()
                },
            }
        )
    return {
        "schema_version": P6_LAYOUT_AUDIT_SCHEMA_VERSION,
        "package": "pyarrow",
        "package_version": str(package_version),
        "layouts": list(layouts),
        "case_count": len(rows),
        "wall_ms": (time.perf_counter() - wall_started) * 1000.0,
        "process_cpu_ms": (time.process_time() - cpu_started) * 1000.0,
        "rows": rows,
    }


def run_seeded_sentinel(backend_audit: dict[str, Any]) -> dict[str, Any]:
    eligible = [
        row
        for row in backend_audit["rows"]
        if row["chdb_ok"] and row["controls_ok"]
    ][:3]
    fault_kinds = ("accept_reject", "row_drop", "value_change")
    rows: list[dict[str, Any]] = []
    for source, fault_kind in zip(eligible, fault_kinds, strict=False):
        original = NormalizedResult.from_dict(source["normalized"]["chdb"])
        faulty = NormalizedResult.from_dict(copy.deepcopy(original.to_dict()))
        if fault_kind == "accept_reject":
            faulty.status = "error"
            faulty.error_type = "SeededSentinelError"
            faulty.rows = []
            faulty.lossless_rows = []
        elif fault_kind == "row_drop":
            faulty.rows = faulty.rows[:-1] if faulty.rows else [["seeded-extra-row"]]
            faulty.lossless_rows = []
        else:
            if faulty.rows and faulty.rows[0]:
                faulty.rows[0][0] = _different_value(faulty.rows[0][0])
            else:
                faulty.rows = [["seeded-value"]]
            faulty.lossless_rows = []
        faulty._comparison_key_cache = ""
        faulty._row_profile_cache = None
        faulty._stable_row_keys_cache = None
        findings = evaluate_case(
            Case.from_dict(dict(source["case"])),
            {"chdb_reference": original, "chdb_seeded_fault": faulty},
            comparison_mode="contract",
        )
        rows.append(
            {
                "case_id": source["case_id"],
                "fault_kind": fault_kind,
                "detected": bool(findings),
                "findings": [finding.to_dict() for finding in findings],
            }
        )
    return {
        "schema_version": P6_SENTINEL_SCHEMA_VERSION,
        "attempted_count": len(rows),
        "detected_count": sum(1 for row in rows if row["detected"]),
        "rows": rows,
    }


def cache_axis_audit(case: Case) -> dict[str, Any]:
    config = frozen_method_config()
    case_digest = ExecutionResultCache.case_digest(case, config)

    pyarrow = PyArrowBackend()
    cache = ExecutionResultCache(
        {"pyarrow": pyarrow},
        environment={"pyarrow": "24.0.0"},
    )
    pyarrow.configure_physical_layout("contiguous")
    contiguous_key = cache.key(
        case_digest=case_digest, backend_name="pyarrow", config=config
    )
    pyarrow.configure_physical_layout("chunked")
    chunked_key = cache.key(
        case_digest=case_digest, backend_name="pyarrow", config=config
    )

    current_cache = ExecutionResultCache(
        {"polars_streaming": make_backend("polars_streaming")},
        environment={"polars": "1.42.1"},
    )
    old_cache = ExecutionResultCache(
        {"polars_streaming": make_backend("polars_streaming")},
        environment={"polars": "1.40.0"},
    )
    current_key = current_cache.key(
        case_digest=case_digest,
        backend_name="polars_streaming",
        config=config,
    )
    old_key = old_cache.key(
        case_digest=case_digest,
        backend_name="polars_streaming",
        config=config,
    )
    lazy_cache = ExecutionResultCache(
        {"polars_lazy": make_backend("polars_lazy")},
        environment={"polars": "1.42.1"},
    )
    lazy_key = lazy_cache.key(
        case_digest=case_digest,
        backend_name="polars_lazy",
        config=config,
    )
    return {
        "schema_version": "p6-cache-axis-audit-v1",
        "layout_keys_distinct": contiguous_key != chunked_key,
        "version_keys_distinct": current_key != old_key,
        "mode_keys_distinct": current_key != lazy_key,
        "keys": {
            "pyarrow_contiguous": contiguous_key,
            "pyarrow_chunked": chunked_key,
            "polars_streaming_1_42_1": current_key,
            "polars_streaming_1_40_0": old_key,
            "polars_lazy_1_42_1": lazy_key,
        },
    }


def summarize_p6(
    *,
    conformance: dict[str, Any],
    backend_audit: dict[str, Any],
    current_mode: dict[str, Any],
    old_mode: dict[str, Any],
    layout_audit: dict[str, Any],
    sentinel: dict[str, Any],
    cache_audit: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    applicable_backend = [row for row in backend_audit["rows"] if row["chdb_applicable"]]
    valid_backend = [row for row in applicable_backend if row["chdb_ok"] and row["controls_ok"]]
    backend_agreement = [row for row in valid_backend if row["agreement"]]

    mode_summaries = {}
    for label, audit in (("current", current_mode), ("second_version", old_mode)):
        valid = [row for row in audit["rows"] if row["valid"]]
        mode_summaries[label] = {
            "version": audit["package_version"],
            "case_count": len(audit["rows"]),
            "valid_count": len(valid),
            "valid_rate": _ratio(len(valid), len(audit["rows"])),
            "agreement_count": sum(1 for row in valid if row["agreement"]),
            "agreement_rate": _ratio(
                sum(1 for row in valid if row["agreement"]), len(valid)
            ),
        }

    layout_summaries = {}
    for layout in ("chunked", "sliced", "dictionary"):
        comparisons = [
            row["comparisons"][layout]
            for row in layout_audit["rows"]
            if layout in row["comparisons"]
        ]
        valid = [row for row in comparisons if row["valid"]]
        layout_summaries[layout] = {
            "applicable_count": len(comparisons),
            "valid_count": len(valid),
            "valid_rate": _ratio(len(valid), len(comparisons)),
            "agreement_count": sum(1 for row in valid if row["agreement"]),
            "agreement_rate": _ratio(
                sum(1 for row in valid if row["agreement"]), len(valid)
            ),
        }

    summary = {
        "adapter_conformance_rate": float(conformance["pass_rate"]),
        "chdb": {
            "case_count": len(backend_audit["rows"]),
            "applicable_count": len(applicable_backend),
            "valid_count": len(valid_backend),
            "valid_rate": _ratio(len(valid_backend), len(applicable_backend)),
            "agreement_count": len(backend_agreement),
            "agreement_rate": _ratio(len(backend_agreement), len(valid_backend)),
            "raw_finding_case_count": sum(
                1 for row in valid_backend if any(row["pair_findings"].values())
            ),
        },
        "polars_modes": mode_summaries,
        "pyarrow_layouts": layout_summaries,
        "seeded_sentinel_detection_rate": _ratio(
            sentinel["detected_count"], sentinel["attempted_count"]
        ),
        "cache_axis_audit": cache_audit,
    }
    gate_results = {
        "adapter_conformance": summary["adapter_conformance_rate"]
        >= float(gates["adapter_conformance_rate_min"]),
        "stable_unsupported_skip": all(
            row["checks"].get("stable_unsupported_skip", False)
            for row in conformance["rows"]
        ),
        "chdb_validity": summary["chdb"]["valid_rate"]
        >= float(gates["chdb_common_capability_valid_rate_min"]),
        "chdb_agreement": summary["chdb"]["agreement_rate"]
        >= float(gates["semantic_agreement_rate_min"]),
        "polars_mode_validity": all(
            row["valid_rate"] >= float(gates["polars_mode_valid_rate_min"])
            for row in mode_summaries.values()
        ),
        "polars_mode_agreement": all(
            row["agreement_rate"] >= float(gates["semantic_agreement_rate_min"])
            for row in mode_summaries.values()
        ),
        "pyarrow_layout_validity": all(
            row["valid_rate"] >= float(gates["pyarrow_layout_valid_rate_min"])
            for row in layout_summaries.values()
        ),
        "pyarrow_layout_agreement": all(
            row["agreement_rate"] >= float(gates["semantic_agreement_rate_min"])
            for row in layout_summaries.values()
        ),
        "seeded_sensitivity": summary["seeded_sentinel_detection_rate"]
        >= float(gates["seeded_sentinel_top_level_detection_rate_min"]),
        "cache_axis_separation": all(
            bool(cache_audit[key])
            for key in (
                "layout_keys_distinct",
                "version_keys_distinct",
                "mode_keys_distinct",
            )
        ),
    }
    return {
        "summary": summary,
        "gates": gate_results,
        "all_gates_passed": all(gate_results.values()),
    }


def _different_value(value: Any) -> Any:
    if value is None:
        return "seeded-non-null"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    return f"{value}-seeded"


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
