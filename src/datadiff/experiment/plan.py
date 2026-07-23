"""Declared, replayable method and campaign plans for paired experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datadiff.experiment_manifest import stable_digest
from datadiff.util import append_jsonl, dump_json, load_json, read_jsonl


METHOD_SPEC_SCHEMA_VERSION = "datadiff-method-spec-v1"
EXPERIMENT_PLAN_SCHEMA_VERSION = "datadiff-experiment-plan-v1"

_DIMENSIONS = ("generation", "scheduling", "execution", "evidence", "semantics")


@dataclass(frozen=True, slots=True)
class MethodSpec:
    method_id: str
    parent_method_id: str
    changed_dimensions: tuple[str, ...]
    generation: dict[str, Any]
    scheduling: dict[str, Any]
    execution: dict[str, Any]
    evidence: dict[str, Any]
    semantics: dict[str, Any]
    arm_kind: str = "research_control"
    schema_version: str = METHOD_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.method_id:
            raise ValueError("method_id must not be empty")
        if self.arm_kind not in {"live_default", "research_control", "candidate"}:
            raise ValueError("unknown method arm kind")
        if self.method_id.startswith("p8_") and self.arm_kind != "candidate":
            raise ValueError("P8 methods must remain candidate arms")
        unknown = set(self.changed_dimensions) - set(_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown changed dimensions: {sorted(unknown)!r}")
        if self.parent_method_id and not self.changed_dimensions:
            raise ValueError("derived methods require explicit changed_dimensions")
        for dimension in _DIMENSIONS:
            value = getattr(self, dimension)
            if not isinstance(value, dict) or not value:
                raise ValueError(f"method dimension {dimension} must be explicit and non-empty")

    @property
    def digest(self) -> str:
        return stable_digest("method-spec", self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_dimensions"] = list(self.changed_dimensions)
        if include_digest:
            payload["digest"] = stable_digest("method-spec", payload)
        return payload

    def changed_from(self, parent: "MethodSpec") -> tuple[str, ...]:
        return tuple(
            dimension
            for dimension in _DIMENSIONS
            if getattr(self, dimension) != getattr(parent, dimension)
        )


@dataclass(frozen=True, slots=True)
class SeedBlock:
    block_id: str
    root_seed: int
    case_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.block_id or not self.case_indices:
            raise ValueError("seed blocks need an id and assigned case indices")
        if len(self.case_indices) != len(set(self.case_indices)):
            raise ValueError("seed block case indices must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {"block_id": self.block_id, "root_seed": self.root_seed, "case_indices": list(self.case_indices)}


@dataclass(frozen=True, slots=True)
class ShardSpec:
    method_id: str
    replicate: int
    seed_block: SeedBlock
    target_shard: tuple[str, ...]
    input_digest: str

    @property
    def shard_id(self) -> str:
        return stable_digest("experiment-shard", self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "replicate": self.replicate,
            "seed_block": self.seed_block.to_dict(),
            "target_shard": list(self.target_shard),
            "input_digest": self.input_digest,
        }


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    plan_id: str
    design: str
    methods: tuple[MethodSpec, ...]
    seed_blocks: tuple[SeedBlock, ...]
    target_shards: tuple[tuple[str, ...], ...]
    replicates: int
    corpus_manifest_digest: str
    resource_budget: dict[str, Any]
    stopping_rule: dict[str, Any]
    primary_metrics: tuple[str, ...]
    secondary_metrics: tuple[str, ...] = ()
    exclusion_rules: tuple[str, ...] = ()
    schema_version: str = EXPERIMENT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.design not in {"paired", "single_factor_ablation", "factorial", "fractional_factorial"}:
            raise ValueError("unknown experiment design")
        if len(self.methods) < 1 or len({item.method_id for item in self.methods}) != len(self.methods):
            raise ValueError("experiment methods must have unique ids")
        if int(self.replicates) <= 0:
            raise ValueError("replicates must be positive")
        if not self.seed_blocks or not self.target_shards:
            raise ValueError("paired plans require seed blocks and target shards")
        if not self.corpus_manifest_digest or not self.resource_budget or not self.stopping_rule:
            raise ValueError("plans require frozen corpus, budget, and stopping rule")
        self.validate_methods()

    @property
    def digest(self) -> str:
        return stable_digest("experiment-plan", self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "design": self.design,
            "methods": [method.to_dict() for method in self.methods],
            "seed_blocks": [block.to_dict() for block in self.seed_blocks],
            "target_shards": [list(shard) for shard in self.target_shards],
            "replicates": self.replicates,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "resource_budget": dict(self.resource_budget),
            "stopping_rule": dict(self.stopping_rule),
            "primary_metrics": list(self.primary_metrics),
            "secondary_metrics": list(self.secondary_metrics),
            "exclusion_rules": list(self.exclusion_rules),
        }
        if include_digest:
            payload["plan_digest"] = stable_digest("experiment-plan", payload)
        return payload

    def validate_methods(self) -> None:
        by_id = {method.method_id: method for method in self.methods}
        for method in self.methods:
            if not method.parent_method_id:
                continue
            parent = by_id.get(method.parent_method_id)
            if parent is None:
                raise ValueError(f"method parent is not part of plan: {method.parent_method_id}")
            actual = method.changed_from(parent)
            if actual != method.changed_dimensions:
                raise ValueError(
                    f"method {method.method_id} has implicit or missing changes: "
                    f"declared={method.changed_dimensions!r}, actual={actual!r}"
                )
            if self.design == "single_factor_ablation" and len(actual) != 1:
                raise ValueError("single-factor ablations must change exactly one dimension")
        if self.design in {"paired", "single_factor_ablation"} and len(self.methods) > 1:
            baselines = [method for method in self.methods if not method.parent_method_id]
            if len(baselines) != 1:
                raise ValueError("paired comparisons require exactly one base method")

    def expand_shards(self) -> list[ShardSpec]:
        output: list[ShardSpec] = []
        for method in sorted(self.methods, key=lambda item: item.method_id):
            for replicate in range(self.replicates):
                for block in sorted(self.seed_blocks, key=lambda item: item.block_id):
                    for target_shard in sorted(self.target_shards):
                        input_payload = {
                            "plan_digest": self.digest,
                            "method_digest": method.digest,
                            "replicate": replicate,
                            "seed_block": block.to_dict(),
                            "target_shard": list(target_shard),
                        }
                        output.append(
                            ShardSpec(
                                method_id=method.method_id,
                                replicate=replicate,
                                seed_block=block,
                                target_shard=target_shard,
                                input_digest=stable_digest("shard-input", input_payload),
                            )
                        )
        return sorted(output, key=lambda item: item.shard_id)


class CampaignArtifacts:
    """Writes the immutable artifact skeleton for a declared campaign."""

    def __init__(self, root: str | Path, plan: ExperimentPlan) -> None:
        self.root = Path(root)
        self.plan = plan
        self.root.mkdir(parents=True, exist_ok=True)

    def initialize(self, *, environment: Mapping[str, Any]) -> None:
        manifest = {
            "schema_version": "experiment-artifacts-v1",
            "plan": self.plan.to_dict(),
            "environment": dict(environment),
            "experiment_manifest_digest": stable_digest(
                "experiment-manifest", {"plan": self.plan.to_dict(), "environment": dict(environment)}
            ),
        }
        manifest_path = self.root / "experiment_manifest.json"
        if manifest_path.exists():
            existing = load_json(manifest_path)
            if existing.get("experiment_manifest_digest") != manifest["experiment_manifest_digest"]:
                raise ValueError("campaign directory belongs to a different immutable manifest")
            return
        dump_json(manifest, manifest_path)
        dump_json(
            {
                "plan_id": self.plan.plan_id,
                "design": self.plan.design,
                "stopping_rule": dict(self.plan.stopping_rule),
                "primary_metrics": list(self.plan.primary_metrics),
                "secondary_metrics": list(self.plan.secondary_metrics),
                "exclusion_rules": list(self.plan.exclusion_rules),
            },
            self.root / "design.json",
        )
        for shard in self.plan.expand_shards():
            append_jsonl(
                {
                    "shard_id": shard.shard_id,
                    "status": "pending",
                    "input_digest": shard.input_digest,
                    "retry_history": [],
                    "shard": shard.to_dict(),
                },
                self.root / "shard_manifest.jsonl",
            )

    def append_metric_row(self, row: Mapping[str, Any]) -> None:
        required = {"seed_block", "method_id"}
        missing = sorted(key for key in required if key not in row)
        if missing:
            raise ValueError(f"metric rows require {missing!r}")
        append_jsonl(dict(row), self.root / "metric_rows.jsonl")

    def shard_records(self) -> dict[str, dict[str, Any]]:
        path = self.root / "shard_manifest.jsonl"
        if not path.exists():
            return {}
        records: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(path):
            shard_id = str(row.get("shard_id", "") or "")
            if shard_id:
                records[shard_id] = row
        return records

    def append_shard_record(self, row: Mapping[str, Any]) -> None:
        required = {"shard_id", "status", "input_digest"}
        missing = sorted(key for key in required if key not in row)
        if missing:
            raise ValueError(f"shard records require {missing!r}")
        append_jsonl(dict(row), self.root / "shard_manifest.jsonl")
