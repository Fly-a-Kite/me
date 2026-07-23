"""Deterministic, process-isolated validation support for semantic novelty.

The scheduler's large-scale protocol is deliberately separate from live
discovery.  It freezes every seed first, starts a fresh worker per shard, and
retains the first failure evidence before any retry is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from datadiff.experiment_manifest import stable_digest


SEMANTIC_NOVELTY_VALIDATION_SCHEMA_VERSION = "semantic-novelty-sharded-validation-v1"


@dataclass(frozen=True, slots=True)
class SemanticNoveltyShardSpec:
    suite: str
    shard: int
    seed: int
    case_budget: int
    targets: tuple[str, ...]
    wall_time_limit_s: float
    rss_limit_mib: int
    backend_call_limit: int

    @property
    def shard_id(self) -> str:
        return f"{self.suite}-shard-{self.shard:03d}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "shard_id": self.shard_id, "targets": list(self.targets)}


def derive_shard_seed(*, suite: str, shard: int, root_seed: int) -> int:
    """Derive a stable seed without consulting time or process randomness."""

    payload = f"semantic-novelty-shard-v1:{int(root_seed)}:{suite}:{int(shard)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def build_seed_manifest(
    *,
    suites: Mapping[str, Sequence[str]],
    root_seed: int,
    shard_count: int = 100,
    case_budget: int = 100,
    wall_time_limit_s: float = 180.0,
    rss_limit_mib: int = 4096,
    backend_call_limit: int | None = None,
) -> dict[str, Any]:
    if shard_count <= 0 or case_budget <= 0:
        raise ValueError("shard_count and case_budget must be positive")
    if wall_time_limit_s <= 0 or rss_limit_mib <= 0:
        raise ValueError("resource limits must be positive")
    rows: list[dict[str, Any]] = []
    for suite, targets in sorted(suites.items()):
        resolved_targets = tuple(str(item) for item in targets if str(item))
        if not suite or not resolved_targets:
            raise ValueError("each suite requires a name and at least one target")
        call_limit = int(backend_call_limit or case_budget * max(1, len(resolved_targets)) * 12)
        for shard in range(shard_count):
            rows.append(
                SemanticNoveltyShardSpec(
                    suite=str(suite),
                    shard=shard,
                    seed=derive_shard_seed(suite=str(suite), shard=shard, root_seed=root_seed),
                    case_budget=int(case_budget),
                    targets=resolved_targets,
                    wall_time_limit_s=float(wall_time_limit_s),
                    rss_limit_mib=int(rss_limit_mib),
                    backend_call_limit=call_limit,
                ).to_dict()
            )
    manifest = {
        "schema_version": SEMANTIC_NOVELTY_VALIDATION_SCHEMA_VERSION,
        "root_seed": int(root_seed),
        "shard_count_per_suite": int(shard_count),
        "case_budget_per_shard": int(case_budget),
        "suites": {name: list(values) for name, values in sorted(suites.items())},
        "shards": rows,
    }
    manifest["manifest_digest"] = stable_digest("semantic-novelty-seed-manifest", manifest)
    return manifest


def shard_spec_from_dict(data: Mapping[str, Any]) -> SemanticNoveltyShardSpec:
    return SemanticNoveltyShardSpec(
        suite=str(data["suite"]),
        shard=int(data["shard"]),
        seed=int(data["seed"]),
        case_budget=int(data["case_budget"]),
        targets=tuple(str(item) for item in data["targets"]),
        wall_time_limit_s=float(data["wall_time_limit_s"]),
        rss_limit_mib=int(data["rss_limit_mib"]),
        backend_call_limit=int(data["backend_call_limit"]),
    )


def aggregate_shard_results(
    manifest: Mapping[str, Any],
    shard_results: Iterable[Mapping[str, Any]],
    *,
    offline_ab_gate: Mapping[str, Any] | None = None,
    blind_holdout_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = {
        str(row["shard_id"])
        for row in manifest.get("shards", ())
        if isinstance(row, Mapping) and row.get("shard_id")
    }
    results = [dict(row) for row in shard_results if isinstance(row, Mapping)]
    completed = {str(row.get("shard_id", "")) for row in results if row.get("status") == "completed"}
    failure_reasons = Counter(
        str(row.get("terminal_reason", ""))
        for row in results
        if str(row.get("status", "")) != "completed"
    )
    depths = [
        float(value)
        for row in results
        for value in row.get("semantic_depths", ()) or ()
        if isinstance(value, (int, float))
    ]
    tokens = sum(int(row.get("unique_semantic_tokens", 0) or 0) for row in results)
    motifs = sum(int(row.get("unique_interaction_motifs", 0) or 0) for row in results)
    families = set().union(*(set(row.get("families", ()) or ()) for row in results)) if results else set()
    lanes = set().union(*(set(row.get("lanes", ()) or ()) for row in results)) if results else set()
    targets = set().union(*(set(row.get("targets", ()) or ()) for row in results)) if results else set()
    candidate_cases = sum(int(row.get("candidate_bug_cases", 0) or 0) for row in results)
    rechecked_candidates = sum(int(row.get("rechecked_candidate_cases", 0) or 0) for row in results)
    fresh_rechecks = sum(int(row.get("fresh_recheck_survivors", 0) or 0) for row in results)
    candidate_families: Counter[str] = Counter()
    for row in results:
        candidate_families.update(
            {
                str(key): int(value or 0)
                for key, value in (row.get("candidate_family_counts", {}) or {}).items()
            }
        )
    confirmed_roots = set().union(*(set(row.get("confirmed_roots", ()) or ()) for row in results)) if results else set()
    cpu_s = sum(float(row.get("cpu_seconds", 0.0) or 0.0) for row in results)
    process_io = Counter()
    for row in results:
        process_io.update({str(key): int(value or 0) for key, value in (row.get("io_delta", {}) or {}).items()})
    resource_failures = sum(
        int(row.get("terminal_reason") in {"wall_time_limit", "rss_limit", "backend_call_limit"})
        for row in results
    )
    harness_errors = sum(int(row.get("case_iteration_failures", 0) or 0) for row in results)
    timeout_count = sum(int(row.get("timeout_count", 0) or 0) for row in results)
    blocked_count = sum(int(row.get("blocked_capability_count", 0) or 0) for row in results)
    summary = {
        "expected_shards": len(expected),
        "completed_shards": len(completed),
        "missing_shards": sorted(expected - completed),
        "failure_reasons": dict(sorted((key, value) for key, value in failure_reasons.items() if key)),
        "unique_semantic_token_sum": tokens,
        "unique_interaction_motif_sum": motifs,
        "family_coverage": sorted(families),
        "lane_coverage": sorted(lanes),
        "target_coverage": sorted(targets),
        "semantic_depth": _depth_summary(depths),
        "duplicate_family_count": sum(int(row.get("duplicate_family_count", 0) or 0) for row in results),
        "saturated_family_count": sum(int(row.get("saturated_family_count", 0) or 0) for row in results),
        "candidate_cases": candidate_cases,
        "candidate_family_counts": dict(sorted(candidate_families.items())),
        "rechecked_candidate_cases": rechecked_candidates,
        "fresh_recheck_survivors": fresh_rechecks,
        "fresh_recheck_survival_rate": fresh_rechecks / rechecked_candidates if rechecked_candidates else 0.0,
        "expected_semantic_divergence_count": sum(
            int(row.get("expected_semantic_divergence_count", 0) or 0) for row in results
        ),
        "needs_manual_confirmation_count": sum(
            int(row.get("needs_manual_confirmation_count", 0) or 0) for row in results
        ),
        "unique_confirmed_roots": len(confirmed_roots),
        "unique_confirmed_roots_per_cpu_hour": len(confirmed_roots) / (cpu_s / 3600.0) if cpu_s else 0.0,
        "wall_seconds": sum(float(row.get("wall_seconds", 0.0) or 0.0) for row in results),
        "cpu_seconds": cpu_s,
        "max_rss_kib": max((int(row.get("max_rss_kib", 0) or 0) for row in results), default=0),
        "io_delta": dict(sorted(process_io.items())),
        "harness_error_count": harness_errors,
        "timeout_count": timeout_count,
        "blocked_capability_count": blocked_count,
        "resource_failure_count": resource_failures,
    }
    gates = {
        "offline_candidate_pool_ab": bool((offline_ab_gate or {}).get("passed", False)),
        "blind_holdout": bool((blind_holdout_gate or {}).get("passed", False)),
        "all_shards_completed": len(completed) == len(expected),
        "no_resource_limit_failure": resource_failures == 0,
        "no_harness_error": harness_errors == 0,
        "semantic_evidence_present": tokens > 0 and motifs > 0,
    }
    return {
        "schema_version": SEMANTIC_NOVELTY_VALIDATION_SCHEMA_VERSION,
        "seed_manifest_digest": str(manifest.get("manifest_digest", "")),
        "summary": summary,
        "gates": gates,
        "promotion_eligible": all(gates.values()),
    }


def _depth_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(mean(ordered), 6),
        "p50": round(_percentile(ordered, 0.50), 6),
        "p95": round(_percentile(ordered, 0.95), 6),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    index = min(len(values) - 1, max(0, math.ceil(len(values) * fraction) - 1))
    return float(values[index])
