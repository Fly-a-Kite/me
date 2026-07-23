from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from datadiff.experiment_manifest import stable_digest


LIFECYCLE_SCHEMA_VERSION = "experiment-lifecycle-v1"


class LifecycleStage(str, Enum):
    CREATED = "created"
    PREPARED = "prepared"
    RAN = "ran"
    FINALIZED = "finalized"
    VERIFIED = "verified"


@dataclass(slots=True)
class ExperimentLifecycle:
    experiment_id: str
    stage: LifecycleStage = LifecycleStage.CREATED
    manifest: dict[str, Any] = field(default_factory=dict)
    raw_result: dict[str, Any] = field(default_factory=dict)
    final_result: dict[str, Any] = field(default_factory=dict)

    def prepare(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        self._require(LifecycleStage.CREATED)
        payload = dict(manifest)
        if not payload.get("manifest_digest"):
            raise ValueError("prepared experiment requires a sealed manifest digest")
        self.manifest = payload
        self.stage = LifecycleStage.PREPARED
        return self.snapshot()

    def run(self, runner: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> dict[str, Any]:
        self._require(LifecycleStage.PREPARED)
        result = dict(runner(dict(self.manifest)))
        result.setdefault("manifest_digest", self.manifest["manifest_digest"])
        if result["manifest_digest"] != self.manifest["manifest_digest"]:
            raise ValueError("run result does not match prepared manifest")
        self.raw_result = result
        self.stage = LifecycleStage.RAN
        return self.snapshot()

    def finalize(
        self,
        finalizer: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> dict[str, Any]:
        self._require(LifecycleStage.RAN)
        result = dict(finalizer(dict(self.raw_result)))
        result.setdefault("raw_result_digest", stable_digest("raw-result", self.raw_result))
        self.final_result = result
        self.stage = LifecycleStage.FINALIZED
        return self.snapshot()

    def verify(self) -> dict[str, Any]:
        self._require(LifecycleStage.FINALIZED)
        expected = stable_digest("raw-result", self.raw_result)
        if self.final_result.get("raw_result_digest") != expected:
            raise ValueError("final analysis is not bound to the raw result")
        self.stage = LifecycleStage.VERIFIED
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        payload = {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "stage": self.stage.value,
            "manifest_digest": str(self.manifest.get("manifest_digest", "") or ""),
            "raw_result_digest": (
                stable_digest("raw-result", self.raw_result) if self.raw_result else ""
            ),
            "final_result_digest": (
                stable_digest("final-result", self.final_result)
                if self.final_result
                else ""
            ),
        }
        payload["lifecycle_digest"] = stable_digest("lifecycle", payload)
        return payload

    def _require(self, expected: LifecycleStage) -> None:
        if self.stage is not expected:
            raise RuntimeError(
                f"experiment lifecycle is {self.stage.value}; expected {expected.value}"
            )
