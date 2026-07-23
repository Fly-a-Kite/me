"""Declared, source-separated campaigns for discovery and seeded sensitivity.

The regular live loop deliberately supports several candidate sources.  This
module makes a source comparison auditable: every source is an explicit method
arm, runs over identical seed blocks, persists raw event sidecars, and refuses
to score a shard whose coordinator reports a mixed source.  Feedback warmup is
retained as a replayable artifact but deliberately excluded from metric rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from datadiff.config import ExperimentConfig
from datadiff.env import collect_environment
from datadiff.evidence import causal_signature_from_row
from datadiff.experiment.campaign import CampaignRunner
from datadiff.experiment.plan import ExperimentPlan, MethodSpec, SeedBlock, ShardSpec
from datadiff.experiment_manifest import stable_digest
from datadiff.util import dump_json, load_json, read_jsonl, run_meta_path


SOURCE_ORDER = ("fresh", "feedback", "semantic", "replay")
SOURCE_REPORT_KEYS = {
    "fresh": "fresh_grammar_generation",
    "feedback": "feedback_mutation",
    "semantic": "semantic_metamorphic_mutation",
    "replay": "known_regression_replay",
}


def build_source_config(
    source: str,
    *,
    generator_profile: str = "common",
    method_arm: str = "contract_lattice_shared_cost_full",
) -> ExperimentConfig:
    """Return a source-pure, low-IO configuration for a scored shard."""

    if source not in SOURCE_REPORT_KEYS:
        raise ValueError(f"unknown source arm: {source}")
    common: dict[str, Any] = {
        "method_arm": method_arm,
        "generator_profile": generator_profile,
        "enable_artifact": False,
        "enable_reducer": False,
        "enable_backend_session_reuse": False,
        "enable_parallel_backend_execution": False,
        "enable_backend_sampling": False,
        "enable_replay_bug": False,
        "candidate_recheck_count": 0,
        "log_level": "minimal",
        "persist_feedback_corpus": False,
    }
    if source == "fresh":
        return ExperimentConfig(
            **common,
            enable_feedback=False,
            fresh_generation_min_share=1.0,
            feedback_mutation_max_share=0.0,
        )
    if source == "feedback":
        return ExperimentConfig(
            **common,
            enable_feedback=True,
            enable_local_source_scheduler=True,
            fresh_generation_min_share=0.0,
            feedback_mutation_max_share=1.0,
        )
    if source == "semantic":
        return ExperimentConfig(
            **common,
            enable_feedback=False,
            enable_semantic_metamorphic_source=True,
            fresh_generation_min_share=0.0,
            feedback_mutation_max_share=1.0,
        )
    return ExperimentConfig(
        **{
            **common,
            "generator_profile": "datafusion_setop_all_duplicate_count",
            "enable_feedback": False,
            # ``enable_replay_bug`` is the existing admission switch for cases
            # whose discovery origin is an explicit historical replay.  It is
            # intentionally true only in this separately reported control arm.
            "enable_replay_bug": True,
            "enable_known_regression_source": True,
            "fresh_generation_min_share": 0.0,
            "feedback_mutation_max_share": 1.0,
        }
    )


def source_seed_manifest(
    *,
    root_seed: int,
    cases: int,
    block_size: int,
    sources: Sequence[str] = SOURCE_ORDER,
    generator_profile: str = "common",
    method_arm: str = "contract_lattice_shared_cost_full",
) -> dict[str, Any]:
    """Freeze source seed assignments without pretending feedback is a corpus."""

    selected = tuple(str(source) for source in sources)
    if not selected or any(source not in SOURCE_REPORT_KEYS for source in selected):
        raise ValueError("source manifest requires known source arms")
    if cases <= 0 or block_size <= 0:
        raise ValueError("cases and block_size must be positive")
    blocks = [
        {
            "block_id": f"block-{start // block_size:03d}",
            "root_seed": int(root_seed) + start,
            "case_indices": list(range(start, min(cases, start + block_size))),
        }
        for start in range(0, int(cases), int(block_size))
    ]
    payload = {
        "schema_version": "datadiff-source-seed-manifest-v1",
        "root_seed": int(root_seed),
        "case_count": int(cases),
        "block_size": int(block_size),
        "sources": list(selected),
        "generator_profile": generator_profile,
        "method_arm": method_arm,
        "blocks": blocks,
        "feedback_warmup_policy": {
            "included_in_scored_metrics": False,
            "description": "Warmup only creates feedback parents; its logs and state are retained per shard.",
        },
    }
    payload["manifest_digest"] = stable_digest("source-seed-manifest", payload)
    return payload


def build_source_plan(
    *,
    plan_id: str,
    seed_manifest: Mapping[str, Any],
    targets: Sequence[str],
    replicates: int = 3,
    sources: Sequence[str] = SOURCE_ORDER,
    generator_profile: str = "common",
    method_arm: str = "contract_lattice_shared_cost_full",
    primary_metrics: Sequence[str] = ("candidate_count", "independent_roots", "wall_ms", "cpu_ms", "rss_mib", "io_bytes"),
) -> ExperimentPlan:
    """Build a paired plan whose only varying method dimension is generation source."""

    selected = tuple(str(source) for source in sources)
    if not selected or selected[0] != "fresh":
        raise ValueError("paired source plans require fresh as the control arm")
    if any(source not in SOURCE_REPORT_KEYS for source in selected):
        raise ValueError("source plan contains an unknown source")
    blocks = tuple(
        SeedBlock(
            block_id=str(block["block_id"]),
            root_seed=int(block["root_seed"]),
            case_indices=tuple(int(index) for index in block["case_indices"]),
        )
        for block in seed_manifest.get("blocks", ())
    )
    if not blocks:
        raise ValueError("source seed manifest has no blocks")
    target_tuple = tuple(str(target) for target in targets if str(target))
    if not target_tuple:
        raise ValueError("source plan requires at least one target")
    methods: list[MethodSpec] = []
    for source in selected:
        config = build_source_config(
            source,
            generator_profile=generator_profile,
            method_arm=method_arm,
        )
        methods.append(
            MethodSpec(
                method_id=f"source_{source}",
                parent_method_id="" if source == "fresh" else "source_fresh",
                changed_dimensions=() if source == "fresh" else ("generation",),
                generation={
                    "source": source,
                    "seed_assignment": "fixed_source_seed_block_v1",
                    "generator_profile": config.generator_profile,
                    "fuzz_config_overrides": config.to_dict(),
                },
                scheduling={
                    "coordinator": "source_separation_v1",
                    "scored_source_must_be_pure": True,
                    "warmup_excluded": True,
                },
                execution={
                    "targets": list(target_tuple),
                    "parallelism": "serial_per_case",
                    "budget": int(seed_manifest["case_count"]),
                    "fuzz_config": {"method_arm": method_arm},
                },
                evidence={
                    "raw_artifacts": "event_journal_content_addressed_sidecar",
                    "candidate_rechecks": 0,
                    "confirmed_root_policy": "not_claimed_by_source_campaign",
                },
                semantics={
                    "normalizer": "lossless-v2",
                    "oracle": "datadiff-oracle-v2",
                    "capability_contract": "target-capability-model-v1",
                    "causal_grouping": "datadiff-causal-signature-v1",
                },
                arm_kind="research_control",
            )
        )
    return ExperimentPlan(
        plan_id=plan_id,
        design="paired",
        methods=tuple(methods),
        seed_blocks=blocks,
        target_shards=(target_tuple,),
        replicates=max(3, int(replicates)),
        corpus_manifest_digest=str(seed_manifest["manifest_digest"]),
        resource_budget={
            "cases_per_method": int(seed_manifest["case_count"]),
            "replicates": max(3, int(replicates)),
            "same_seed_blocks": True,
        },
        stopping_rule={"kind": "fixed_seed_blocks", "max_blocks": len(blocks)},
        primary_metrics=tuple(str(metric) for metric in primary_metrics),
        secondary_metrics=(
            "candidate_causal_signature_count",
            "confirmed_root_count",
            "fault_detection_rate",
            "coverage_cells",
            "harness_error_rate",
        ),
        exclusion_rules=(
            "feedback_warmup_excluded_from_all_scored_metrics",
            "known_replay_excluded_from_fresh_only_conclusions",
            "seeded_fault_excluded_from_real_target_discovery_conclusions",
            "candidate_is_not_confirmed_root",
        ),
    )


class SourceSeparatedShardExecutor:
    """Execute source-pure run-loop shards without process-global path mutation."""

    def __init__(
        self,
        root: str | Path,
        *,
        feedback_warmup_cases: int = 8,
    ) -> None:
        self.root = Path(root)
        self.feedback_warmup_cases = max(1, int(feedback_warmup_cases))

    def __call__(self, shard: ShardSpec, method: MethodSpec) -> Mapping[str, Any]:
        from datadiff import runner

        source = str(method.generation.get("source", "") or "")
        expected_source = SOURCE_REPORT_KEYS.get(source)
        if expected_source is None:
            raise ValueError(f"method does not declare a source arm: {method.method_id}")
        payload = method.generation.get("fuzz_config_overrides")
        if not isinstance(payload, Mapping):
            raise ValueError("source method lacks explicit fuzz_config_overrides")
        config = ExperimentConfig.from_payload(payload)
        shard_root = self.root / "source_runs" / shard.shard_id
        scored_root = shard_root / "scored"
        seed = int(shard.seed_block.root_seed)
        case_count = len(shard.seed_block.case_indices)
        backends = list(shard.target_shard)
        warmup: dict[str, Any] | None = None
        closed_loop_state: dict[str, Any] | None = None
        if source == "feedback":
            warmup_file = runner.run_fuzz(
                cases=self.feedback_warmup_cases,
                seed=seed - 10_000,
                backends=backends,
                config=config,
                persist_closed_loop_state=True,
                runs_dir=shard_root / "warmup" / "runs",
                corpus_dir=shard_root / "warmup" / "corpus",
            )
            warmup_meta = load_json(run_meta_path(warmup_file))
            state_path = Path(str(warmup_meta.get("closed_loop_state_file", "") or ""))
            if not state_path.is_file():
                raise ValueError("feedback warmup did not persist closed-loop state")
            closed_loop_state = load_json(state_path)
            _bias_feedback_scheduler(closed_loop_state)
            warmup = {
                "excluded_from_scored_metrics": True,
                "run_file": _relative_to_root(warmup_file, self.root),
                "run_meta": _relative_to_root(run_meta_path(warmup_file), self.root),
                "closed_loop_state": _relative_to_root(state_path, self.root),
                "executed_cases": int(warmup_meta.get("executed_cases", 0) or 0),
                "coordinator": dict(warmup_meta.get("coordinator", {}) or {}),
            }
        run_file = runner.run_fuzz(
            cases=case_count,
            seed=seed,
            backends=backends,
            config=config,
            closed_loop_state=closed_loop_state,
            persist_closed_loop_state=True,
            runs_dir=scored_root / "runs",
            corpus_dir=scored_root / "corpus",
        )
        meta = load_json(run_meta_path(run_file))
        coordinator = dict(meta.get("coordinator", {}) or {})
        source_reports = _validate_source_purity(
            coordinator,
            expected_source=expected_source,
            expected_cases=case_count,
        )
        rows = read_jsonl(run_file)
        metrics = _scored_metrics(rows, meta, expected_source=expected_source, backends=backends)
        metrics["source_reports"] = source_reports
        return {
            "schema_version": "datadiff-source-separated-shard-v1",
            "source": source,
            "expected_source": expected_source,
            "shard": shard.to_dict(),
            "scored_run": {
                "run_file": _relative_to_root(run_file, self.root),
                "run_meta": _relative_to_root(run_meta_path(run_file), self.root),
                "event_evidence_dir": _relative_to_root(
                    Path(str(meta.get("event_evidence_dir", "") or "")), self.root
                ),
                "closed_loop_state": _relative_to_root(
                    Path(str(meta.get("closed_loop_state_file", "") or "")), self.root
                ),
            },
            "warmup": warmup,
            "coordinator": coordinator,
            "metrics": metrics,
        }


def run_source_campaign(
    root: str | Path,
    plan: ExperimentPlan,
    *,
    feedback_warmup_cases: int = 8,
    workers: int = 1,
) -> dict[str, Any]:
    """Run a source plan and emit the standard immutable experiment artifacts."""

    return CampaignRunner(
        root,
        plan,
        environment=collect_environment(),
        execute_shard=SourceSeparatedShardExecutor(
            root,
            feedback_warmup_cases=feedback_warmup_cases,
        ),
        workers=max(1, int(workers)),
    ).run()


def write_source_preregistration(
    root: str | Path,
    *,
    plan: ExperimentPlan,
    seed_manifest: Mapping[str, Any],
    fault_target: str = "",
) -> None:
    """Persist the no-ambiguity protocol that accompanies a source campaign."""

    payload = {
        "schema_version": "datadiff-source-separated-preregistration-v1",
        "hypothesis": (
            "Candidate sources are measured separately on fixed seed blocks; "
            "warmup and known replay never enter fresh-only conclusions."
        ),
        "fault_target": fault_target,
        "fault_target_policy": (
            "Controlled injected fault only; not a real-target discovery result."
            if fault_target
            else "No seeded fault is present in this campaign."
        ),
        "paired_unit": "(replicate, seed_block, target_shard)",
        "seed_manifest": dict(seed_manifest),
        "plan": plan.to_dict(),
        "scored_metrics": list(plan.primary_metrics),
        "exclusion_rules": list(plan.exclusion_rules),
        "confirmation_policy": (
            "Causal-signature roots are candidates only. Independent process replay, "
            "minimization, and target/version recording are required before confirmation."
        ),
    }
    dump_json(payload, Path(root) / "preregistration.json")


def _bias_feedback_scheduler(state: dict[str, Any]) -> None:
    """Record a deterministic feedback-only scored phase after retained warmup."""

    scheduler = state.get("source_scheduler")
    if not isinstance(scheduler, dict):
        return
    arms = scheduler.setdefault("arms", {})
    generated = arms.setdefault("generated", {})
    feedback = arms.setdefault("feedback_mutation", {})
    generated.update({"pulls": 100, "total_reward": -250.0})
    feedback.update({"pulls": 100, "total_reward": 250.0})
    scheduler["total_pulls"] = 200
    feedback_state = state.get("feedback")
    if isinstance(feedback_state, dict):
        feedback_state["source_scheduler"] = scheduler


def _validate_source_purity(
    coordinator: Mapping[str, Any],
    *,
    expected_source: str,
    expected_cases: int,
) -> dict[str, dict[str, Any]]:
    reports = coordinator.get("source_reports", {})
    if not isinstance(reports, Mapping):
        raise ValueError("scored run lacks coordinator source reports")
    normalized = {
        str(source): dict(report)
        for source, report in reports.items()
        if isinstance(report, Mapping)
    }
    actual = int(normalized.get(expected_source, {}).get("cases", 0) or 0)
    mixed = {
        source: int(report.get("cases", 0) or 0)
        for source, report in normalized.items()
        if source != expected_source and int(report.get("cases", 0) or 0)
    }
    if actual != expected_cases or mixed:
        raise ValueError(
            "source separation violated: "
            f"expected {expected_source}={expected_cases}, actual={actual}, mixed={mixed}"
        )
    return {source: normalized.get(source, _empty_source_report()) for source in sorted(SOURCE_REPORT_KEYS.values())}


def _scored_metrics(
    rows: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    *,
    expected_source: str,
    backends: Sequence[str],
) -> dict[str, Any]:
    candidate_cases = 0
    harness_errors = 0
    signatures: set[str] = set()
    fault_backend = next((backend for backend in backends if backend.startswith("buggy_")), "")
    fault_detections = 0
    for row in rows:
        findings = [finding for finding in row.get("findings", ()) if isinstance(finding, Mapping)]
        candidate_findings = [
            finding
            for finding in findings
            if str(finding.get("triage_verdict", "") or "") == "candidate_implementation_bug"
        ]
        candidate_cases += int(bool(candidate_findings))
        harness_errors += int(
            any(
                str(finding.get("triage_verdict", "") or "") == "harness_lowering_error"
                for finding in findings
            )
        )
        for finding in candidate_findings:
            for backend in finding.get("suspicious_backends", ()) or ():
                if str(backend):
                    signatures.add(causal_signature_from_row(row, finding, backend=str(backend)).group_key)
            if fault_backend and fault_backend in {str(item) for item in finding.get("suspicious_backends", ()) or ()}:
                fault_detections += 1
                break
    count = len(rows)
    wall = _nested_number(meta, "wall_time_profile", "totals_ms", "total_wall_ms")
    cpu = _nested_number(meta, "process_cpu_profile", "totals_ms", "total_process_cpu_ms")
    return {
        "case_count": count,
        "candidate_count": candidate_cases,
        "candidate_causal_signature_count": len(signatures),
        "causal_signatures": sorted(signatures),
        "independent_roots": len(signatures),
        "confirmed_root_count": 0,
        "fault_detection_count": fault_detections,
        "fault_detection_rate": fault_detections / count if fault_backend and count else 0.0,
        "harness_error_rate": harness_errors / count if count else 0.0,
        "wall_ms": wall,
        "cpu_ms": cpu,
        "rss_mib": _max_rss_mib(),
        "io_bytes": _artifact_bytes(meta),
        "scored_source": expected_source,
    }


def _nested_number(payload: Mapping[str, Any], *path: str) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping):
            return 0.0
        value = value.get(key)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _artifact_bytes(meta: Mapping[str, Any]) -> float:
    paths = [
        Path(value)
        for value in (
            str(meta.get("run_file", "") or ""),
            str(meta.get("event_evidence_dir", "") or ""),
            str(meta.get("closed_loop_state_file", "") or ""),
        )
        if value
    ]
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return float(total)


def _max_rss_mib() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes while Linux reports KiB.
        return value / (1024.0 * 1024.0) if value > 1024.0 * 1024.0 else value / 1024.0
    except (ImportError, OSError, ValueError):
        return 0.0


def _relative_to_root(path: Path, root: Path) -> str:
    if not path:
        return ""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _empty_source_report() -> dict[str, Any]:
    return {"cases": 0, "candidates": 0, "independent_root_count": 0, "cpu_ms": 0.0, "io_bytes": 0.0}
