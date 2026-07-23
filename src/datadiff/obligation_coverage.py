from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

from datadiff.ccs_ablation import load_frozen_cases, validate_frozen_case_corpus
from datadiff.ccs_ir import case_to_ccs_ir
from datadiff.experiment_manifest import stable_digest
from datadiff.metamorphic import all_metamorphic_variants
from datadiff.test_obligations import (
    executable_obligation_registry,
    select_executable_test_obligations,
)
from datadiff.util import utc_now


OBLIGATION_COVERAGE_AUDIT_SCHEMA_VERSION = "ccs-obligation-coverage-audit-v2"


def audit_obligation_coverage(
    corpus: Mapping[str, Any],
    *,
    perf_counter_fn: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    validate_frozen_case_corpus(corpus)
    registry = executable_obligation_registry()
    registered_relations = set(registry["relations"])
    relation_rows: dict[str, dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    all_construction_ms: list[float] = []
    guided_construction_ms: list[float] = []
    category_counts: Counter[str] = Counter()

    for case_index, case in enumerate(load_frozen_cases(corpus)):
        started = perf_counter_fn()
        legacy_variants = all_metamorphic_variants(case)
        all_construction_ms.append((perf_counter_fn() - started) * 1000.0)
        ir = case_to_ccs_ir(case)
        node_declared_relations, program_declared_relations = (
            _declared_relations_by_scope(ir)
        )
        declared_relations = node_declared_relations | program_declared_relations
        started = perf_counter_fn()
        selection = select_executable_test_obligations(case, ir)
        guided_construction_ms.append((perf_counter_fn() - started) * 1000.0)

        legacy_keys = {(variant.relation, variant.name) for variant in legacy_variants}
        guided_keys = {(variant.relation, variant.name) for variant in selection.variants}
        legacy_sequence = [
            (variant.relation, variant.name) for variant in legacy_variants
        ]
        guided_sequence = [
            (variant.relation, variant.name) for variant in selection.variants
        ]
        legacy_relations = {relation for relation, _name in legacy_keys}
        guided_relations = {relation for relation, _name in guided_keys}
        declared_registered = declared_relations & registered_relations
        not_declared = legacy_relations - declared_relations
        declared_not_registered = declared_relations - registered_relations
        registered_not_constructed = declared_registered - guided_relations
        constructed_not_legacy = guided_relations - legacy_relations

        for relation in sorted(
            legacy_relations | declared_relations | guided_relations | registered_relations
        ):
            row = relation_rows.setdefault(
                relation,
                {
                    "relation": relation,
                    "registered": relation in registered_relations,
                    "legacy_applicable_case_count": 0,
                    "legacy_variant_count": 0,
                    "declared_case_count": 0,
                    "node_declared_case_count": 0,
                    "program_declared_case_count": 0,
                    "guided_constructed_case_count": 0,
                    "guided_variant_count": 0,
                    "missing_declaration_case_ids": [],
                    "not_constructible_case_ids": [],
                },
            )
            relation_legacy_count = sum(
                1 for variant in legacy_variants if variant.relation == relation
            )
            relation_guided_count = sum(
                1 for variant in selection.variants if variant.relation == relation
            )
            if relation in legacy_relations:
                row["legacy_applicable_case_count"] += 1
                row["legacy_variant_count"] += relation_legacy_count
            if relation in declared_relations:
                row["declared_case_count"] += 1
            if relation in node_declared_relations:
                row["node_declared_case_count"] += 1
            if relation in program_declared_relations:
                row["program_declared_case_count"] += 1
            if relation in guided_relations:
                row["guided_constructed_case_count"] += 1
                row["guided_variant_count"] += relation_guided_count
            if relation in not_declared:
                row["missing_declaration_case_ids"].append(case.case_id)
            if relation in registered_not_constructed:
                row["not_constructible_case_ids"].append(case.case_id)

        category_counts.update(
            {
                "legacy_variant_instances": len(legacy_keys),
                "guided_variant_instances": len(guided_keys),
                "covered_legacy_variant_instances": len(legacy_keys & guided_keys),
                "legacy_relation_opportunities": len(legacy_relations),
                "covered_legacy_relation_opportunities": len(
                    legacy_relations & guided_relations
                ),
                "declared_relation_opportunities": len(declared_relations),
                "node_declared_relation_opportunities": len(
                    node_declared_relations
                ),
                "program_declared_relation_opportunities": len(
                    program_declared_relations
                ),
                "registered_declared_opportunities": len(declared_registered),
                "constructible_declared_opportunities": len(
                    declared_registered & guided_relations
                ),
                "exact_variant_order_match_cases": int(
                    legacy_sequence == guided_sequence
                ),
            }
        )
        case_rows.append(
            {
                "case_index": case_index,
                "case_id": case.case_id,
                "seed": case.seed,
                "legacy_variant_count": len(legacy_keys),
                "guided_variant_count": len(guided_keys),
                "legacy_relations": sorted(legacy_relations),
                "declared_relations": sorted(declared_relations),
                "node_declared_relations": sorted(node_declared_relations),
                "program_declared_relations": sorted(
                    program_declared_relations
                ),
                "guided_relations": sorted(guided_relations),
                "legacy_relations_not_declared": sorted(not_declared),
                "declared_relations_not_registered": sorted(
                    declared_not_registered
                ),
                "registered_relations_not_constructed": sorted(
                    registered_not_constructed
                ),
                "constructed_relations_not_in_legacy_all": sorted(
                    constructed_not_legacy
                ),
                "variant_instance_recall": _ratio(
                    len(legacy_keys & guided_keys), len(legacy_keys)
                ),
                "relation_opportunity_recall": _ratio(
                    len(legacy_relations & guided_relations), len(legacy_relations)
                ),
                "exact_variant_order_match": legacy_sequence == guided_sequence,
                "obligation_selection": selection.to_dict(),
            }
        )

    legacy_relation_names = {
        relation
        for row in case_rows
        for relation in row["legacy_relations"]
    }
    guided_relation_names = {
        relation
        for row in case_rows
        for relation in row["guided_relations"]
    }
    payload = {
        "schema_version": OBLIGATION_COVERAGE_AUDIT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "corpus_digest": str(corpus.get("corpus_digest", "")),
        "case_count": len(case_rows),
        "registry_digest": registry["digest"],
        "registered_relation_count": len(registered_relations),
        "summary": {
            **dict(category_counts),
            "variant_instance_recall": _ratio(
                category_counts["covered_legacy_variant_instances"],
                category_counts["legacy_variant_instances"],
            ),
            "relation_opportunity_recall": _ratio(
                category_counts["covered_legacy_relation_opportunities"],
                category_counts["legacy_relation_opportunities"],
            ),
            "declaration_registration_rate": _ratio(
                category_counts["registered_declared_opportunities"],
                category_counts["declared_relation_opportunities"],
            ),
            "registered_obligation_constructibility_rate": _ratio(
                category_counts["constructible_declared_opportunities"],
                category_counts["registered_declared_opportunities"],
            ),
            "unique_legacy_relation_count": len(legacy_relation_names),
            "unique_guided_relation_count": len(guided_relation_names),
            "unique_relation_recall": _ratio(
                len(legacy_relation_names & guided_relation_names),
                len(legacy_relation_names),
            ),
            "exact_variant_order_match_rate": _ratio(
                category_counts["exact_variant_order_match_cases"],
                len(case_rows),
            ),
            "legacy_all_construction_ms": _timing_summary(all_construction_ms),
            "ccs_guided_construction_ms": _timing_summary(guided_construction_ms),
        },
        "missing_legacy_relations": sorted(
            legacy_relation_names - guided_relation_names
        ),
        "guided_only_relations": sorted(
            guided_relation_names - legacy_relation_names
        ),
        "relations": [relation_rows[key] for key in sorted(relation_rows)],
        "cases": case_rows,
    }
    payload["audit_digest"] = stable_digest("obligation-coverage", payload)
    return payload


def render_obligation_coverage_markdown(audit: Mapping[str, Any]) -> str:
    summary = audit.get("summary", {})
    lines = [
        "# CCS Obligation Coverage Audit",
        "",
        f"- Corpus digest: `{audit.get('corpus_digest', '')}`",
        f"- Cases: {audit.get('case_count', 0)}",
        f"- Registry digest: `{audit.get('registry_digest', '')}`",
        f"- Variant-instance recall: {_format_rate(summary.get('variant_instance_recall'))}",
        f"- Relation-opportunity recall: {_format_rate(summary.get('relation_opportunity_recall'))}",
        f"- Unique relation recall: {_format_rate(summary.get('unique_relation_recall'))}",
        f"- Exact default variant-order match: {_format_rate(summary.get('exact_variant_order_match_rate'))}",
        f"- Declaration registration: {_format_rate(summary.get('declaration_registration_rate'))}",
        f"- Registered obligation constructibility: {_format_rate(summary.get('registered_obligation_constructibility_rate'))}",
        "",
        "## Missing Legacy Relations",
        "",
    ]
    missing = list(audit.get("missing_legacy_relations", ()))
    lines.extend(f"- `{relation}`" for relation in missing)
    if not missing:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Per-Relation Coverage",
            "",
            "| Relation | Registered | Legacy cases | Node declarations | Program declarations | Guided cases | Legacy variants | Guided variants |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for relation in audit.get("relations", ()):
        lines.append(
            "| `{relation}` | {registered} | {legacy} | {node_declared} | {program_declared} | {guided} | {legacy_variants} | {guided_variants} |".format(
                relation=relation.get("relation", ""),
                registered="yes" if relation.get("registered") else "no",
                legacy=relation.get("legacy_applicable_case_count", 0),
                node_declared=relation.get("node_declared_case_count", 0),
                program_declared=relation.get("program_declared_case_count", 0),
                guided=relation.get("guided_constructed_case_count", 0),
                legacy_variants=relation.get("legacy_variant_count", 0),
                guided_variants=relation.get("guided_variant_count", 0),
            )
        )
    return "\n".join(lines) + "\n"


def _declared_relations_by_scope(ir) -> tuple[set[str], set[str]]:
    node_relations = {
        obligation.obligation_id
        for node in ir.nodes
        for obligation in node.contract.test_obligations
        if obligation.kind == "metamorphic_relation"
    }
    program_relations = {
        obligation.obligation_id for obligation in ir.program_obligations
    }
    return node_relations, program_relations


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "total_ms": sum(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "max_ms": max(values, default=0.0),
    }


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"
