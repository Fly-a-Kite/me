from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Mapping

from datadiff.classification_oracle import annotate_findings
from datadiff.config import ExperimentConfig
from datadiff.dsl import Case
from datadiff.metamorphic import evaluate_metamorphic_variants
from datadiff.normalizer import NormalizedResult
from datadiff.oracle import Finding, evaluate_case
from datadiff.oracle_complex import cross_validate_oracle_findings
from datadiff.run_findings import _countable_row_findings


def derive_factorial_candidate_row(
    case: Case,
    full_row: Mapping[str, Any],
    *,
    backends: Iterable[str],
    variant_names: Iterable[str],
    config: ExperimentConfig,
    evaluate_case_fn: Callable[..., list[Finding]] = evaluate_case,
    evaluate_metamorphic_variants_fn: Callable[..., list[Finding]] = (
        evaluate_metamorphic_variants
    ),
    annotate_findings_fn: Callable[..., None] = annotate_findings,
) -> dict[str, Any]:
    """Slice one full factorial execution into a conditional candidate arm.

    Backend results are never re-executed here. This is valid only for
    candidate-event estimands; fresh reproduction remains a separate action.
    """

    if config.enable_witness_oracle:
        raise ValueError("factorial candidate derivation does not support witness oracle")
    started = time.perf_counter()
    names = tuple(dict.fromkeys(str(value) for value in backends))
    if not names:
        raise ValueError("factorial derivation requires at least one backend")
    full_normalized = full_row.get("normalized", {})
    full_raw = full_row.get("raw_results", {})
    if not isinstance(full_normalized, Mapping) or not isinstance(full_raw, Mapping):
        raise ValueError("full factorial row lacks base execution payloads")
    missing_backends = [name for name in names if name not in full_normalized]
    if missing_backends:
        raise ValueError(f"full factorial row lacks backends: {missing_backends}")
    normalized = {
        name: NormalizedResult.from_dict(full_normalized[name], backend=name)
        for name in names
    }
    raw_results = {
        name: dict(full_raw[name])
        for name in names
        if name in full_raw and isinstance(full_raw[name], Mapping)
    }
    differential_findings = (
        evaluate_case_fn(case, normalized)
        if config.enable_differential_oracle
        else []
    )

    requested_variants = tuple(dict.fromkeys(str(value) for value in variant_names))
    full_metamorphic = full_row.get("metamorphic", {})
    if not isinstance(full_metamorphic, Mapping):
        raise ValueError("full factorial row has invalid metamorphic payload")
    missing_variants = [
        name for name in requested_variants if name not in full_metamorphic
    ]
    if missing_variants:
        raise ValueError(
            f"full factorial row lacks requested variants: {missing_variants}"
        )
    variant_results: dict[str, dict[str, NormalizedResult]] = {}
    metamorphic_rows: dict[str, Any] = {}
    for variant_name in requested_variants:
        source = full_metamorphic[variant_name]
        if not isinstance(source, Mapping):
            raise ValueError("full factorial variant row is invalid")
        source_normalized = source.get("normalized", {})
        if not isinstance(source_normalized, Mapping):
            raise ValueError("full factorial variant lacks normalized results")
        variant_results[variant_name] = {
            name: NormalizedResult.from_dict(source_normalized[name], backend=name)
            for name in names
            if name in source_normalized
        }
        source_raw = source.get("raw_results", {})
        metamorphic_rows[variant_name] = {
            "relation": str(source.get("relation", "")),
            "case": source.get("case", {}),
            "raw_results": {
                name: source_raw[name]
                for name in names
                if isinstance(source_raw, Mapping) and name in source_raw
            },
            "normalized": {
                name: result.to_dict()
                for name, result in variant_results[variant_name].items()
            },
        }
    metamorphic_findings = (
        evaluate_metamorphic_variants_fn(case, normalized, variant_results)
        if config.enable_metamorphic_oracle
        else []
    )
    findings = [*differential_findings, *metamorphic_findings]
    if findings:
        annotate_findings_fn(
            case,
            findings,
            normalized=normalized,
            raw_results=raw_results,
            config=config.to_dict(),
            backends=list(names),
        )
    finding_payloads = [finding.to_dict() for finding in findings]
    countable = _countable_row_findings({"findings": finding_payloads})
    target_rows = full_row.get("targets", []) or []
    targets = [
        row
        for row in target_rows
        if isinstance(row, Mapping) and str(row.get("name", "")) in names
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "schema_version": "rlcmf-factorial-derived-candidate-row-v1",
        "derivation_scope": "candidate_events_without_fresh_reproduction",
        "case": case.to_dict(),
        "targets": targets,
        "raw_results": raw_results,
        "normalized": {name: result.to_dict() for name, result in normalized.items()},
        "metamorphic": metamorphic_rows,
        "metamorphic_selection": {
            "executed_variants": list(requested_variants),
            "executed_relations": list(
                dict.fromkeys(
                    str(row.get("relation", ""))
                    for row in metamorphic_rows.values()
                    if str(row.get("relation", ""))
                )
            ),
            "variant_limit": len(requested_variants),
            "factorial_derived": True,
        },
        "oracle_cross_validation": cross_validate_oracle_findings(
            differential_findings,
            metamorphic_findings,
        ).to_dict(),
        "findings": finding_payloads,
        "candidate_recheck": {
            "enabled": False,
            "attempts": 0,
            "reproduced_keys": [],
            "non_reproduced_keys": [],
            "skip_reason": "factorial_candidate_derivation_requires_separate_recheck",
        },
        "config": config.to_dict(),
        "status": "bug" if countable else "ok",
        "duration_ms": elapsed_ms,
        "stage_profile": {
            "oracle_classification_ms": elapsed_ms,
            "total_case_wall_ms": elapsed_ms,
        },
        "execution_profile": {
            "factorial_derived": True,
            "backend_reported_total_ms": 0.0,
            "backend_count": len(names),
            "backend_calls": 0,
        },
    }
