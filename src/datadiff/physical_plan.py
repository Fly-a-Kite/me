from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any, Literal

from datadiff.plan_fingerprint import fingerprint_plan


PHYSICAL_PLAN_COLLECTOR_SCHEMA_VERSION = "physical-plan-collector-v2"
PHYSICAL_PLAN_COLLECTOR_VERSION = "p8-tiered-collector-1"

PlanDetail = Literal["fingerprint", "full"]

PlanTextSupplier = Callable[[], Any]
ClockFn = Callable[[], float]


@dataclass(frozen=True, slots=True)
class PlanObservation:
    plan_kind: str
    status: str
    raw_text: str = ""
    normalized_text: str = ""
    fingerprint: str = ""
    operator_tokens: tuple[str, ...] = ()
    error_type: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_kind": self.plan_kind,
            "status": self.status,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "fingerprint": self.fingerprint,
            "operator_tokens": list(self.operator_tokens),
            "error_type": self.error_type,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class PhysicalPlanBundle:
    backend: str
    backend_version: str
    status: str
    observations: tuple[PlanObservation, ...]
    collection_duration_ms: float
    collector_version: str = PHYSICAL_PLAN_COLLECTOR_VERSION
    unsupported_reason: str = ""
    detail: PlanDetail = "full"

    @property
    def fingerprints(self) -> tuple[str, ...]:
        return tuple(
            observation.fingerprint
            for observation in self.observations
            if observation.status == "ok" and observation.fingerprint
        )

    @property
    def operator_tokens(self) -> tuple[str, ...]:
        return tuple(
            token
            for observation in self.observations
            if observation.status == "ok"
            for token in observation.operator_tokens
        )

    def observation(self, plan_kind: str) -> PlanObservation | None:
        return next(
            (
                observation
                for observation in self.observations
                if observation.plan_kind == plan_kind
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PHYSICAL_PLAN_COLLECTOR_SCHEMA_VERSION,
            "collector_version": self.collector_version,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "status": self.status,
            "collection_duration_ms": self.collection_duration_ms,
            "unsupported_reason": self.unsupported_reason,
            "detail": self.detail,
            "observations": [
                observation.to_dict() for observation in self.observations
            ],
        }


def collect_physical_plan_bundle(
    *,
    backend: str,
    backend_version: str,
    sources: Mapping[str, PlanTextSupplier],
    unsupported: Mapping[str, str] | None = None,
    detail: PlanDetail = "full",
    clock_fn: ClockFn = time.perf_counter,
) -> PhysicalPlanBundle:
    """Collect plan text without allowing observation failure to fail execution.

    Suppliers must inspect an already-built query/lazy frame. They must not execute
    the case result a second time. Each source is isolated so one unavailable plan
    kind produces a partial bundle instead of discarding the other observations.
    """

    if detail not in {"fingerprint", "full"}:
        raise ValueError(f"unsupported physical plan detail: {detail}")
    started = clock_fn()
    observations: list[PlanObservation] = []
    for plan_kind, supplier in sources.items():
        try:
            raw_text = str(supplier() or "")
            if not raw_text.strip():
                raise ValueError("collector returned empty plan text")
            fingerprint = fingerprint_plan(
                raw_text,
                backend=backend,
                plan_kind=str(plan_kind),
            )
            observations.append(
                PlanObservation(
                    plan_kind=str(plan_kind),
                    status="ok",
                    raw_text=raw_text if detail == "full" else "",
                    normalized_text=(
                        fingerprint.normalized_text if detail == "full" else ""
                    ),
                    fingerprint=fingerprint.fingerprint,
                    operator_tokens=fingerprint.operator_tokens,
                )
            )
        except Exception as exc:  # noqa: BLE001 - collector is fail-closed
            observations.append(
                PlanObservation(
                    plan_kind=str(plan_kind),
                    status="error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
    for plan_kind, reason in (unsupported or {}).items():
        if any(item.plan_kind == str(plan_kind) for item in observations):
            raise ValueError(f"duplicate plan kind: {plan_kind}")
        observations.append(
            PlanObservation(
                plan_kind=str(plan_kind),
                status="unsupported",
                error_type="UnsupportedPlanKind",
                error=str(reason),
            )
        )
    duration_ms = max(0.0, (clock_fn() - started) * 1000.0)
    ok_count = sum(observation.status == "ok" for observation in observations)
    if not observations:
        status = "unsupported"
    elif ok_count == len(observations):
        status = "ok"
    elif ok_count:
        status = "partial"
    else:
        status = "error"
    return PhysicalPlanBundle(
        backend=str(backend),
        backend_version=str(backend_version or ""),
        status=status,
        observations=tuple(observations),
        collection_duration_ms=duration_ms,
        unsupported_reason=("no plan sources registered" if not observations else ""),
        detail=detail,
    )


def unsupported_physical_plan_bundle(
    *,
    backend: str,
    backend_version: str = "",
    reason: str,
) -> PhysicalPlanBundle:
    return PhysicalPlanBundle(
        backend=str(backend),
        backend_version=str(backend_version or ""),
        status="unsupported",
        observations=(),
        collection_duration_ms=0.0,
        unsupported_reason=str(reason),
        detail="fingerprint",
    )
