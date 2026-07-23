"""Execution engine for declared, resumable experiment plans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable

from datadiff.experiment.plan import CampaignArtifacts, ExperimentPlan, MethodSpec, ShardSpec
from datadiff.experiment_manifest import stable_digest
from datadiff.coordinator import merge_coordinator_snapshots
from datadiff.evidence import ContentAddressedSidecar
from datadiff.util import append_jsonl, dump_json, read_jsonl


ShardExecutor = Callable[[ShardSpec, MethodSpec], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ShardExecution:
    shard: ShardSpec
    status: str
    output: dict[str, Any]
    retry_history: tuple[dict[str, str], ...]


class CampaignRunner:
    """Run the immutable shard matrix, then emit deterministic analysis.

    The executor is deliberately injected: a production launcher can use
    isolated worker processes while tests and small studies can use a local
    callable.  The plan, shard input digest, retry history and merge order are
    owned here, so neither launcher can silently swap corpus blocks or seeds.
    """

    def __init__(
        self,
        root: str | Path,
        plan: ExperimentPlan,
        *,
        environment: Mapping[str, Any],
        execute_shard: ShardExecutor,
        max_retries: int = 0,
        workers: int = 1,
    ) -> None:
        self.artifacts = CampaignArtifacts(root, plan)
        self.plan = plan
        self.environment = dict(environment)
        self.execute_shard = execute_shard
        self.max_retries = max(0, int(max_retries))
        self.workers = max(1, int(workers))

    def run(self) -> dict[str, Any]:
        self.artifacts.initialize(environment=self.environment)
        methods = {method.method_id: method for method in self.plan.methods}
        previous = self.artifacts.shard_records()
        pending = [
            shard
            for shard in self.plan.expand_shards()
            if not _is_reusable_success(
                previous.get(shard.shard_id),
                shard,
                root=self.artifacts.root,
            )
        ]
        metric_shards = _metric_shard_ids(self.artifacts.root / "metric_rows.jsonl")
        for execution in self._execute_pending(pending, methods, previous):
            self._write_execution(execution, metric_shards)
        return self.finalize()

    def finalize(self) -> dict[str, Any]:
        rows = _sorted_metric_rows(self.artifacts.root / "metric_rows.jsonl")
        analysis = _analyze(self.plan, rows)
        status_counts: dict[str, int] = defaultdict(int)
        for record in self.artifacts.shard_records().values():
            status_counts[str(record.get("status", "unknown") or "unknown")] += 1
        expected_shards = len(self.plan.expand_shards())
        completed = status_counts.get("completed", 0)
        analysis["completion"] = {
            "expected_shards": expected_shards,
            "completed_shards": completed,
            "failed_shards": status_counts.get("failed", 0),
            "pending_shards": status_counts.get("pending", 0),
            "status_counts": dict(sorted(status_counts.items())),
            "complete": completed == expected_shards and not status_counts.get("failed", 0),
        }
        analysis["analysis_digest"] = stable_digest(
            "campaign-analysis",
            {key: value for key, value in analysis.items() if key != "analysis_digest"},
        )
        dump_json(analysis, self.artifacts.root / "analysis.json")
        (self.artifacts.root / "report.md").write_text(
            _report_markdown(self.plan, analysis), encoding="utf-8"
        )
        return analysis

    def _execute_pending(
        self,
        shards: list[ShardSpec],
        methods: Mapping[str, MethodSpec],
        previous: Mapping[str, Mapping[str, Any]],
    ) -> Iterable[ShardExecution]:
        def execute(shard: ShardSpec) -> ShardExecution:
            return self._execute_one(shard, methods[shard.method_id], previous.get(shard.shard_id, {}))

        if self.workers == 1 or len(shards) < 2:
            for shard in shards:
                yield execute(shard)
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(execute, shard) for shard in shards]
            for future in as_completed(futures):
                yield future.result()

    def _execute_one(
        self,
        shard: ShardSpec,
        method: MethodSpec,
        previous: Mapping[str, Any],
    ) -> ShardExecution:
        history = [dict(item) for item in previous.get("retry_history", ()) if isinstance(item, Mapping)]
        for attempt in range(self.max_retries + 1):
            try:
                output = dict(self.execute_shard(shard, method))
                return ShardExecution(shard, "completed", output, tuple(history))
            except Exception as exc:  # noqa: BLE001 - failures are shard evidence, not lost work
                history.append({"attempt": str(len(history) + 1), "error": f"{type(exc).__name__}: {exc}"})
        return ShardExecution(shard, "failed", {}, tuple(history))

    def _write_execution(self, execution: ShardExecution, metric_shards: set[str]) -> None:
        shard = execution.shard
        output_digest = stable_digest("shard-output", execution.output)
        output_ref = ""
        if execution.status == "completed":
            output_path = self.artifacts.root / "shard_results" / f"{shard.shard_id}.json"
            dump_json(execution.output, output_path)
            output_ref = output_path.relative_to(self.artifacts.root).as_posix()
        record = {
            "shard_id": shard.shard_id,
            "status": execution.status,
            "input_digest": shard.input_digest,
            "output_digest": output_digest if execution.status == "completed" else "",
            "output_ref": output_ref,
            "retry_history": list(execution.retry_history),
            "shard": shard.to_dict(),
        }
        self.artifacts.append_shard_record(record)
        if execution.status != "completed" or shard.shard_id in metric_shards:
            return
        row = _metric_row(shard, execution.output, output_digest)
        self.artifacts.append_metric_row(row)
        metric_shards.add(shard.shard_id)


def run_campaign(
    root: str | Path,
    plan: ExperimentPlan,
    *,
    environment: Mapping[str, Any],
    execute_shard: ShardExecutor,
    max_retries: int = 0,
    workers: int = 1,
) -> dict[str, Any]:
    return CampaignRunner(
        root,
        plan,
        environment=environment,
        execute_shard=execute_shard,
        max_retries=max_retries,
        workers=workers,
    ).run()


class FrozenCorpusShardExecutor:
    """Run one declared shard against exactly its assigned frozen cases.

    It is the production bridge between an :class:`ExperimentPlan` and the
    existing trusted ``run_loaded_case`` path.  No seed is generated here:
    shard indices select immutable corpus entries, which preserves paired
    comparison blocks across methods and retries.
    """

    def __init__(
        self,
        corpus: Mapping[str, Any],
        *,
        environment: Mapping[str, Any],
        artifact_root: str | Path,
        run_case_fn: Callable[..., Mapping[str, Any]] | None = None,
        describe_targets_fn: Callable[[list[str]], list[dict[str, Any]]] | None = None,
    ) -> None:
        from datadiff.ccs_ablation import validate_frozen_case_corpus
        from datadiff.runner import run_loaded_case
        from datadiff.targets import describe_targets

        validate_frozen_case_corpus(corpus)
        self.corpus = dict(corpus)
        self.environment = dict(environment)
        self.artifact_root = Path(artifact_root)
        self.sidecars = ContentAddressedSidecar(artifact_root)
        self.run_case_fn = run_case_fn or run_loaded_case
        self.describe_targets_fn = describe_targets_fn or describe_targets

    def __call__(self, shard: ShardSpec, method: MethodSpec) -> Mapping[str, Any]:
        from datadiff.dsl import Case
        from datadiff.campaign_control import capability_units_for_case, record_coordinator_outcome
        from datadiff.coordinator import CandidateSource, Coordinator, ScheduledCase

        config = _fuzz_config_for_method(method)
        target_shard = list(shard.target_shard)
        targets = self.describe_targets_fn(target_shard)
        target_capabilities = [
            set(target.get("capabilities", ()) or ())
            for target in targets
            if isinstance(target, Mapping)
        ]
        common_capabilities = set.intersection(*target_capabilities) if target_capabilities else set()
        coordinator = Coordinator(common_capabilities)
        corpus_cases = self.corpus.get("cases", ())
        rows: list[dict[str, Any]] = []
        for case_index in shard.seed_block.case_indices:
            if case_index < 0 or case_index >= len(corpus_cases):
                raise ValueError(f"seed block references corpus case {case_index}, outside the frozen corpus")
            case = Case.from_dict(dict(corpus_cases[case_index]))
            self._record_case_progress(shard, case_index, case.case_id, status="started")
            row = dict(
                self.run_case_fn(
                    case,
                    backends=target_shard,
                    config=config,
                    save_artifact=False,
                    environment=self.environment,
                    target_specs=targets,
                )
            )
            rows.append(row)
            self._record_case_progress(shard, case_index, case.case_id, status="completed")
            scheduled = ScheduledCase(
                case_id=case.case_id,
                case_index=case_index,
                source=CandidateSource.FRESH_GRAMMAR,
                derived_seed=case.seed,
                target_shard=",".join(target_shard),
                capability_units=capability_units_for_case(
                    case,
                    target_shard=",".join(target_shard),
                ),
            )
            coordinator.admit(scheduled)
            record_coordinator_outcome(
                coordinator,
                scheduled,
                row=row,
                causal_signatures=_row_causal_signatures(row),
                witness_id=stable_digest(
                    "frozen-case-witness",
                    {"corpus": self.corpus["corpus_digest"], "case_index": case_index},
                ),
            )
        coordinator_snapshot = coordinator.snapshot()
        metrics = _frozen_corpus_metrics(
            rows,
            coordinator_snapshot=coordinator_snapshot,
            grouping_mode=str(
                method.semantics.get(
                    "causal_grouping",
                    "datadiff-causal-signature-v1",
                )
            ),
        )
        return {
            "schema_version": "datadiff-frozen-corpus-shard-v2",
            "corpus_digest": str(self.corpus["corpus_digest"]),
            "method_id": method.method_id,
            "method_digest": method.digest,
            "shard": shard.to_dict(),
            "config": config.to_dict(),
            "case_rows": [_compact_case_row(row, sidecars=self.sidecars) for row in rows],
            "coordinator": coordinator_snapshot,
            "metrics": metrics,
        }

    def _record_case_progress(
        self,
        shard: ShardSpec,
        case_index: int,
        case_id: str,
        *,
        status: str,
    ) -> None:
        append_jsonl(
            {
                "schema_version": "datadiff-frozen-case-progress-v1",
                "shard_id": shard.shard_id,
                "case_index": case_index,
                "case_id": case_id,
                "status": status,
            },
            self.artifact_root / "case_progress.jsonl",
        )


def run_frozen_campaign(
    root: str | Path,
    plan: ExperimentPlan,
    corpus: Mapping[str, Any],
    *,
    environment: Mapping[str, Any],
    max_retries: int = 0,
    workers: int = 1,
    run_case_fn: Callable[..., Mapping[str, Any]] | None = None,
    describe_targets_fn: Callable[[list[str]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Execute a declared plan on a verified corpus and write all artifacts."""

    from datadiff.ccs_ablation import validate_frozen_case_corpus

    validate_frozen_case_corpus(corpus)
    corpus_digest = str(corpus.get("corpus_digest", "") or "")
    if corpus_digest != plan.corpus_manifest_digest:
        raise ValueError("experiment plan and frozen corpus digest do not match")
    executor = FrozenCorpusShardExecutor(
        corpus,
        environment=environment,
        artifact_root=root,
        run_case_fn=run_case_fn,
        describe_targets_fn=describe_targets_fn,
    )
    return run_campaign(
        root,
        plan,
        environment=environment,
        execute_shard=executor,
        max_retries=max_retries,
        workers=workers,
    )


def _compact_case_row(
    row: Mapping[str, Any],
    *,
    sidecars: ContentAddressedSidecar,
) -> dict[str, Any]:
    """Keep shard JSON bounded while preserving lossless raw observations."""

    compact = dict(row)
    raw_results = compact.get("raw_results", {})
    if isinstance(raw_results, Mapping):
        compact["raw_result_refs"] = {
            str(backend): sidecars.put("raw-result", raw).to_dict()
            for backend, raw in raw_results.items()
            if isinstance(raw, Mapping)
        }
        compact["raw_results"] = {
            str(backend): _raw_result_summary(raw)
            for backend, raw in raw_results.items()
            if isinstance(raw, Mapping)
        }
    case = compact.pop("case", None)
    if isinstance(case, Mapping):
        compact["case_ref"] = {
            "case_id": str(case.get("case_id", "") or ""),
            "digest": stable_digest("case", case),
        }
    return compact


def _raw_result_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: raw[key]
        for key in (
            "status",
            "error_type",
            "error",
            "duration_ms",
            "cpu_ms",
            "rss_mib",
            "io_read_bytes",
            "io_write_bytes",
        )
        if key in raw
    }


def _fuzz_config_for_method(method: MethodSpec):
    """Resolve only explicitly declared fuzz configuration for an arm."""

    from datadiff.config import ExperimentConfig

    base = method.execution.get("fuzz_config")
    if not isinstance(base, Mapping) or not base:
        raise ValueError("frozen-corpus methods require execution.fuzz_config")
    payload = dict(base)
    for dimension in (
        method.generation,
        method.scheduling,
        method.execution,
        method.evidence,
        method.semantics,
    ):
        overrides = dimension.get("fuzz_config_overrides")
        if overrides is None:
            continue
        if not isinstance(overrides, Mapping):
            raise ValueError("fuzz_config_overrides must be a mapping")
        payload.update(overrides)
    return ExperimentConfig.from_payload(payload)


def _frozen_corpus_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    coordinator_snapshot: Mapping[str, Any] | None = None,
    grouping_mode: str = "datadiff-causal-signature-v1",
) -> dict[str, Any]:
    from datadiff.harness_errors import harness_lowering_errors

    roots: set[str] = set()
    causal_roots: set[str] = set()
    broad_roots: set[str] = set()
    broad_to_causal: dict[str, set[str]] = defaultdict(set)
    capability_cells: set[str] = set()
    candidate_count = false_positive_count = undecided_count = harness_error_count = 0
    wall_ms = cpu_ms = io_bytes = 0.0
    rss_mib = 0.0
    for row in rows:
        findings = [item for item in row.get("findings", ()) if isinstance(item, Mapping)]
        raw_results = row.get("raw_results", {})
        raw_results = raw_results if isinstance(raw_results, Mapping) else {}
        harness_error_count += int(bool(harness_lowering_errors(raw_results)))
        for finding in findings:
            verdict = str(finding.get("triage_verdict", "") or "")
            if verdict == "candidate_implementation_bug":
                candidate_count += 1
                broad_keys = _row_broad_group_keys(finding)
                causal_keys = _row_causal_signatures(row, finding=finding)
                broad_roots.update(broad_keys)
                causal_roots.update(causal_keys)
                for broad_key in broad_keys:
                    broad_to_causal[broad_key].update(causal_keys)
                roots.update(
                    broad_keys
                    if grouping_mode == "broad-semantic-tag-v1"
                    else causal_keys
                )
            if verdict in {"generator_false_positive", "normalizer_false_positive", "harness_lowering_error"}:
                false_positive_count += 1
            if verdict in {"needs_manual_confirmation", "semantic_divergence_needs_confirmation", "unclassified"}:
                undecided_count += 1
        ir = row.get("program_ir", {})
        if isinstance(ir, Mapping):
            capability_cells.update(str(item) for item in ir.get("required_capabilities", ()) if str(item))
        wall_ms += _metric_number(row, "wall_ms")
        cpu_ms += _metric_number(row, "cpu_ms")
        for raw in raw_results.values():
            if not isinstance(raw, Mapping):
                continue
            io_bytes += _number(raw.get("io_read_bytes", 0.0)) + _number(raw.get("io_write_bytes", 0.0))
            rss_mib = max(rss_mib, _number(raw.get("rss_mib", 0.0)))
    case_count = len(rows)
    coordinator = dict(coordinator_snapshot or {})
    coverage = coordinator.get("coverage", {}) if isinstance(coordinator.get("coverage", {}), Mapping) else {}
    readiness = coordinator.get("generalization_readiness", {})
    broad_collision_groups = sum(
        1 for causal_keys in broad_to_causal.values() if len(causal_keys) > 1
    )
    broad_merged_causal_roots = sum(
        max(0, len(causal_keys) - 1) for causal_keys in broad_to_causal.values()
    )
    return {
        "case_count": case_count,
        "candidate_count": candidate_count,
        "independent_roots": len(roots),
        "causal_signatures": sorted(roots),
        "root_group_keys": sorted(roots),
        "root_grouping_mode": grouping_mode,
        "causal_root_count": len(causal_roots),
        "broad_root_count": len(broad_roots),
        "broad_collision_group_count": broad_collision_groups,
        "broad_merged_causal_root_count": broad_merged_causal_roots,
        "root_group_collision_count": (
            broad_collision_groups
            if grouping_mode == "broad-semantic-tag-v1"
            else 0
        ),
        "coverage_cells": int(coverage.get("cell_count", len(capability_cells)) or 0),
        "coverage_debt": int(coverage.get("debt_cells", 0) or 0),
        "coverage_status_counts": dict(coverage.get("status_counts", {}) or {}),
        "generalization_readiness": dict(readiness) if isinstance(readiness, Mapping) else {},
        "false_positive_rate": false_positive_count / case_count if case_count else 0.0,
        "undecided_rate": undecided_count / case_count if case_count else 0.0,
        "harness_error_rate": harness_error_count / case_count if case_count else 0.0,
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "rss_mib": rss_mib,
        "io_bytes": io_bytes,
        "source_reports": coordinator.get("source_reports") or {
            "fresh_grammar_generation": {
                "cases": case_count,
                "candidates": candidate_count,
                "independent_root_count": len(roots),
                "cpu_ms": cpu_ms,
                "io_bytes": io_bytes,
            }
        },
    }


def _row_causal_signatures(
    row: Mapping[str, Any],
    *,
    finding: Mapping[str, Any] | None = None,
) -> list[str]:
    from datadiff.evidence import causal_signature_from_row

    findings = [finding] if finding is not None else [
        item
        for item in row.get("findings", ())
        if isinstance(item, Mapping)
        and str(item.get("triage_verdict", "") or "") == "candidate_implementation_bug"
    ]
    return sorted(
        {
            causal_signature_from_row(row, item, backend=str(backend)).group_key
            for item in findings
            for backend in item.get("suspicious_backends", ()) or ()
            if str(backend)
        }
    )


def _row_broad_group_keys(finding: Mapping[str, Any]) -> list[str]:
    """Legacy root grouping that intentionally ignores execution identity."""

    broad = str(
        finding.get("root_cause", "")
        or finding.get("semantic_tag", "")
        or finding.get("kind", "unknown")
    )
    return [f"broad-semantic-tag:{broad}"]


def _is_reusable_success(
    record: Mapping[str, Any] | None,
    shard: ShardSpec,
    *,
    root: Path,
) -> bool:
    output_ref = str((record or {}).get("output_ref", "") or "")
    output_path = root / output_ref
    return bool(
        record
        and record.get("status") == "completed"
        and record.get("input_digest") == shard.input_digest
        and record.get("output_digest")
        and output_ref
        and output_path.is_file()
    )


def _metric_row(shard: ShardSpec, output: Mapping[str, Any], output_digest: str) -> dict[str, Any]:
    metrics = output.get("metrics", output)
    metric_data = dict(metrics) if isinstance(metrics, Mapping) else {}
    row = {
        "shard_id": shard.shard_id,
        "method_id": shard.method_id,
        "replicate": shard.replicate,
        "seed_block": shard.seed_block.block_id,
        "root_seed": shard.seed_block.root_seed,
        "case_indices": list(shard.seed_block.case_indices),
        "target_shard": list(shard.target_shard),
        "input_digest": shard.input_digest,
        "output_digest": output_digest,
        **metric_data,
    }
    coordinator = output.get("coordinator")
    if isinstance(coordinator, Mapping):
        row["coordinator"] = dict(coordinator)
    return row


def _metric_shard_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("shard_id", "") or "") for row in read_jsonl(path) if row.get("shard_id")}


def _sorted_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return sorted(
        (dict(row) for row in read_jsonl(path)),
        key=lambda row: (
            str(row.get("method_id", "")),
            int(row.get("replicate", 0) or 0),
            str(row.get("seed_block", "")),
            tuple(str(value) for value in row.get("target_shard", ()) or ()),
            str(row.get("shard_id", "")),
        ),
    )


def _analyze(plan: ExperimentPlan, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method_id", ""))].append(row)
    summary = {
        method.method_id: _method_summary(by_method.get(method.method_id, []))
        for method in sorted(plan.methods, key=lambda item: item.method_id)
    }
    analysis = {
        "schema_version": "datadiff-campaign-analysis-v1",
        "plan_digest": plan.digest,
        "metric_row_count": len(rows),
        "methods": summary,
        "paired_effects": _paired_effects(plan, by_method),
    }
    analysis["analysis_digest"] = stable_digest("campaign-analysis", analysis)
    return analysis


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resources = {
        name: _distribution([_metric_number(row, name) for row in rows])
        for name in ("wall_ms", "cpu_ms", "rss_mib", "io_bytes")
    }
    sources: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        source_reports = row.get("source_reports", {})
        if not isinstance(source_reports, Mapping):
            continue
        for source, report in source_reports.items():
            if not isinstance(report, Mapping):
                continue
            for metric in ("cases", "candidates", "independent_root_count", "cpu_ms", "io_bytes"):
                sources[str(source)][metric] += _number(report.get(metric, 0.0))
    signatures = {
        str(signature)
        for row in rows
        for signature in (
            row.get("root_group_keys", ())
            or row.get("causal_signatures", ())
            or ()
        )
        if str(signature)
    }
    root_count = len(signatures) or int(
        sum(_metric_number(row, "independent_roots") for row in rows)
    )
    cpu_ms = resources["cpu_ms"]["sum"]
    io_bytes = resources["io_bytes"]["sum"]
    snapshots = [
        coordinator
        for row in rows
        if isinstance((coordinator := row.get("coordinator")), Mapping)
    ]
    merged_coverage = merge_coordinator_snapshots(snapshots) if snapshots else {}
    readiness = merged_coverage.get("generalization_readiness", {})
    readiness = dict(readiness) if isinstance(readiness, Mapping) else {}
    unresolved_units = [str(unit) for unit in readiness.get("unresolved_units", ()) if str(unit)]
    return {
        "seed_block_count": len({str(row.get("seed_block", "")) for row in rows}),
        "shard_count": len(rows),
        "primary_metrics": {
            name: _distribution([_metric_number(row, name) for row in rows])
            for name in ("independent_roots", "candidate_count", "coverage_cells", "false_positive_rate", "undecided_rate", "harness_error_rate")
        },
        "resources": resources,
        "cost_curve": {
            "independent_root_count": root_count,
            "roots_per_cpu_hour": root_count / (cpu_ms / 3_600_000.0) if cpu_ms else 0.0,
            "roots_per_gb_io": root_count / (io_bytes / 1_000_000_000.0) if io_bytes else 0.0,
        },
        "generalization": {
            "eligible": bool(readiness.get("eligible", False)),
            "ready_shard_count": sum(
                bool((row.get("generalization_readiness", {}) or {}).get("eligible", False))
                for row in rows
            ),
            "unresolved_cell_count": len(unresolved_units),
            "unresolved_units": unresolved_units,
            "coverage": dict(merged_coverage.get("coverage", {}) or {}),
            "coverage_bitmap": dict(merged_coverage.get("coverage_bitmap", {}) or {}),
            "minimum_witnesses": list(merged_coverage.get("minimum_witnesses", ()) or ()),
            "merge": dict(merged_coverage.get("merge", {}) or {}),
        },
        "source_reports": {source: dict(values) for source, values in sorted(sources.items())},
    }


def _paired_effects(
    plan: ExperimentPlan,
    by_method: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for method in sorted(plan.methods, key=lambda item: item.method_id):
        if not method.parent_method_id:
            continue
        parent_rows = {
            _pair_key(row): row for row in by_method.get(method.parent_method_id, [])
        }
        treatment_rows = {
            _pair_key(row): row for row in by_method.get(method.method_id, [])
        }
        for metric in plan.primary_metrics:
            deltas = [
                _metric_number(treatment_rows[key], metric) - _metric_number(parent_rows[key], metric)
                for key in sorted(set(parent_rows) & set(treatment_rows))
            ]
            interval = _bootstrap_ci(deltas, seed=f"{plan.digest}:{method.method_id}:{metric}")
            effects.append(
                {
                    "control": method.parent_method_id,
                    "treatment": method.method_id,
                    "metric": metric,
                    "paired_block_count": len(deltas),
                    "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
                    "bootstrap_95_ci": interval,
                    "directional_judgment": _directional_judgment(
                        interval,
                        higher_is_better=_higher_is_better(metric),
                    ),
                }
            )
    return effects


def _pair_key(row: Mapping[str, Any]) -> tuple[int, str, tuple[str, ...]]:
    return (
        int(row.get("replicate", 0) or 0),
        str(row.get("seed_block", "")),
        tuple(str(value) for value in row.get("target_shard", ()) or ()),
    )


def _metric_number(row: Mapping[str, Any], name: str) -> float:
    aliases = {
        "wall_ms": ("wall_ms", ("wall_time_profile", "total_wall_ms")),
        "cpu_ms": ("cpu_ms", ("process_cpu_profile", "total_process_cpu_ms")),
        "rss_mib": ("rss_mib",),
        "io_bytes": ("io_bytes",),
        "candidate_count": ("candidate_count", "candidates"),
        "coverage_cells": ("coverage_cells", "coverage_debt"),
        "independent_roots": ("independent_roots", "independent_root_count"),
    }
    for path in aliases.get(name, (name,)):
        value: Any
        if isinstance(path, tuple):
            value = row
            found = True
            for key in path:
                if not isinstance(value, Mapping) or key not in value:
                    found = False
                    break
                value = value[key]
            if not found:
                continue
        else:
            if path not in row:
                continue
            value = row[path]
        if value not in ({}, None):
            return _number(value)
    return 0.0


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "sum": sum(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else 0.0,
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _bootstrap_ci(values: list[float], *, seed: str) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = Random(int(stable_digest("bootstrap", seed).split("-")[-1][:16], 16))
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(1_000)
    )
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _higher_is_better(metric: str) -> bool:
    """State the outcome direction before interpreting a paired interval."""

    return metric not in {
        "wall_ms",
        "cpu_ms",
        "rss_mib",
        "io_bytes",
        "false_positive_rate",
        "undecided_rate",
        "harness_error_rate",
    }


def _directional_judgment(interval: Sequence[float], *, higher_is_better: bool = True) -> str:
    if len(interval) != 2:
        return "inconclusive"
    if higher_is_better:
        if interval[0] > 0.0:
            return "treatment_better"
        if interval[1] < 0.0:
            return "control_better"
    else:
        if interval[1] < 0.0:
            return "treatment_better"
        if interval[0] > 0.0:
            return "control_better"
    return "inconclusive"


def _number(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _report_markdown(plan: ExperimentPlan, analysis: Mapping[str, Any]) -> str:
    lines = [
        f"# Campaign {plan.plan_id}",
        "",
        f"Plan digest: `{plan.digest}`",
        "",
    ]
    completion = analysis.get("completion", {})
    if completion:
        lines.extend(
            [
                f"Shard completion: `{completion.get('completed_shards', 0)}/{completion.get('expected_shards', 0)}`; "
                f"complete={bool(completion.get('complete', False))}.",
                "",
            ]
        )
    lines.extend(
        [
        "## Method summary",
        "",
        "| Method | Shards | Seed blocks | Wall p50/p95 (ms) | CPU p50/p95 (ms) | RSS p50/p95 (MiB) | IO p50/p95 (bytes) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_id, summary in analysis.get("methods", {}).items():
        resources = summary.get("resources", {})
        wall = resources.get("wall_ms", {})
        cpu = resources.get("cpu_ms", {})
        rss = resources.get("rss_mib", {})
        io_bytes = resources.get("io_bytes", {})
        lines.append(
            f"| {method_id} | {summary.get('shard_count', 0)} | {summary.get('seed_block_count', 0)} | "
            f"{wall.get('p50', 0.0):.3f}/{wall.get('p95', 0.0):.3f} | {cpu.get('p50', 0.0):.3f}/{cpu.get('p95', 0.0):.3f} | "
            f"{rss.get('p50', 0.0):.3f}/{rss.get('p95', 0.0):.3f} | {io_bytes.get('p50', 0.0):.3f}/{io_bytes.get('p95', 0.0):.3f} |"
        )
    lines.extend(["", "## Source and cost breakdown", ""])
    for method_id, summary in analysis.get("methods", {}).items():
        cost = summary.get("cost_curve", {})
        lines.append(
            f"- `{method_id}`: {cost.get('independent_root_count', 0)} independent roots; "
            f"{cost.get('roots_per_cpu_hour', 0.0):.3f}/CPU-hour; "
            f"{cost.get('roots_per_gb_io', 0.0):.3f}/GB IO."
        )
        generalization = summary.get("generalization", {})
        coverage = generalization.get("coverage", {})
        lines.append(
            f"  - merged generalization eligible={bool(generalization.get('eligible', False))}; "
            f"unresolved capability cells={generalization.get('unresolved_cell_count', 0)}; "
            f"witnessed={coverage.get('status_counts', {}).get('witnessed', 0)}; "
            f"unsupported-with-evidence={coverage.get('status_counts', {}).get('unsupported-with-evidence', 0)}; "
            f"blocked={coverage.get('status_counts', {}).get('blocked', 0)}."
        )
        for source, report in summary.get("source_reports", {}).items():
            lines.append(
                f"  - `{source}`: cases={report.get('cases', 0):.0f}, "
                f"candidates={report.get('candidates', 0):.0f}, "
                f"roots={report.get('independent_root_count', 0):.0f}."
            )
    lines.extend(["", "## Paired effects", "", "| Control | Treatment | Metric | Blocks | Mean delta | 95% bootstrap CI |", "|---|---|---|---:|---:|---:|"])
    for effect in analysis.get("paired_effects", []):
        ci = effect.get("bootstrap_95_ci", [0.0, 0.0])
        lines.append(
            f"| {effect['control']} | {effect['treatment']} | {effect['metric']} | {effect['paired_block_count']} | "
            f"{effect['mean_delta']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] ({effect['directional_judgment']}) |"
        )
    return "\n".join(lines) + "\n"
