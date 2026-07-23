from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from datadiff.risk_limiting_audit import AUDIT_AXES, _validated_sha256


COVERAGE_STRATUM_DIMENSIONS = (
    "seed",
    "backend",
    "relation",
    "generator",
    "workload",
    "mode",
    "layout",
    "temperature",
)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class CoverageStratum:
    """Canonical fine-grained coverage coordinates for one omitted item."""

    dimensions: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized = {}
        for key, value in self.dimensions.items():
            dimension = str(key or "").strip()
            item = str(value or "").strip()
            if dimension not in COVERAGE_STRATUM_DIMENSIONS:
                raise ValueError(f"unknown coverage stratum dimension: {dimension}")
            if not item:
                raise ValueError("coverage stratum values cannot be empty")
            normalized[dimension] = item
        if not normalized:
            raise ValueError("coverage stratum requires at least one dimension")
        object.__setattr__(self, "dimensions", dict(sorted(normalized.items())))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CoverageStratum":
        return cls(
            dimensions={
                str(key): str(value)
                for key, value in payload.items()
                if value is not None and str(value).strip()
            }
        )

    @property
    def key(self) -> str:
        return "|".join(
            f"{dimension}={self.dimensions[dimension]}"
            for dimension in COVERAGE_STRATUM_DIMENSIONS
            if dimension in self.dimensions
        )

    def with_item(self, dimension: str, value: str) -> "CoverageStratum":
        updated = dict(self.dimensions)
        updated[str(dimension)] = str(value)
        return CoverageStratum(updated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-coverage-stratum-v1",
            "key": self.key,
            "dimensions": dict(self.dimensions),
        }


@dataclass(frozen=True, slots=True)
class CoverageDebtPolicy:
    axis: str
    stratum: str
    max_age_cases: int
    max_outstanding: int

    def __post_init__(self) -> None:
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown debt axis: {self.axis}")
        if not str(self.stratum or "").strip():
            raise ValueError("debt stratum cannot be empty")
        if int(self.max_age_cases) <= 0:
            raise ValueError("max_age_cases must be positive")
        if int(self.max_outstanding) <= 0:
            raise ValueError("max_outstanding must be positive")

    @property
    def key(self) -> tuple[str, str]:
        return (self.axis, self.stratum)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CoverageDebtEntry:
    debt_id: str
    axis: str
    stratum: str
    item_key: str
    created_case: int
    estimated_units: float
    debt_kind: str
    exact_replayable: bool
    metadata: dict[str, Any]
    status: str = "outstanding"
    settled_case: int | None = None
    settlement_reason: str = ""
    audit_decision_id: str = ""

    def __post_init__(self) -> None:
        _validated_sha256(self.debt_id, field="debt_id")
        if self.axis not in AUDIT_AXES:
            raise ValueError(f"unknown debt axis: {self.axis}")
        if not self.stratum or not self.item_key:
            raise ValueError("debt stratum and item_key cannot be empty")
        if self.created_case < 0:
            raise ValueError("created_case must be non-negative")
        if not math.isfinite(self.estimated_units) or self.estimated_units <= 0:
            raise ValueError("estimated_units must be positive")
        if self.debt_kind not in {"quota", "exact"}:
            raise ValueError("debt_kind must be quota or exact")
        if self.debt_kind == "exact" and not self.exact_replayable:
            raise ValueError("exact debt must be replayable")
        if self.status not in {"outstanding", "settled"}:
            raise ValueError("unknown debt status")
        if self.status == "settled" and self.settled_case is None:
            raise ValueError("settled debt requires settled_case")
        if self.status == "outstanding" and self.settled_case is not None:
            raise ValueError("outstanding debt cannot have settled_case")
        if self.status == "settled":
            if int(self.settled_case) < self.created_case:
                raise ValueError("settled_case cannot precede created_case")
            if not str(self.settlement_reason or "").strip():
                raise ValueError("settled debt requires a settlement reason")
        elif self.settlement_reason or self.audit_decision_id:
            raise ValueError("outstanding debt cannot have settlement evidence")
        if self.audit_decision_id:
            _validated_sha256(
                self.audit_decision_id,
                field="audit_decision_id",
            )

    def age(self, current_case: int) -> int:
        return max(0, int(current_case) - self.created_case)

    def to_dict(self) -> dict[str, Any]:
        return {
            "debt_id": self.debt_id,
            "axis": self.axis,
            "stratum": self.stratum,
            "item_key": self.item_key,
            "created_case": self.created_case,
            "estimated_units": self.estimated_units,
            "debt_kind": self.debt_kind,
            "exact_replayable": self.exact_replayable,
            "metadata": dict(self.metadata),
            "status": self.status,
            "settled_case": self.settled_case,
            "settlement_reason": self.settlement_reason,
            "audit_decision_id": self.audit_decision_id,
        }


@dataclass(frozen=True, slots=True)
class ForcedSettlement:
    debt_id: str
    axis: str
    stratum: str
    item_key: str
    reason: str
    age_cases: int
    exact_replayable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoverageDebtLedger:
    """Bounded omission debt with deterministic, restart-safe ordering."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        policies: Iterable[CoverageDebtPolicy],
    ) -> None:
        self.manifest_sha256 = _validated_sha256(
            manifest_sha256,
            field="manifest_sha256",
        )
        policy_items = tuple(policies)
        self.policies = {policy.key: policy for policy in policy_items}
        if not self.policies:
            raise ValueError("at least one coverage-debt policy is required")
        if len(self.policies) != len(policy_items):
            raise ValueError("duplicate coverage-debt policies are not allowed")
        self.entries: dict[str, CoverageDebtEntry] = {}
        self._next_serial = 0

    def _policy(self, axis: str, stratum: str) -> CoverageDebtPolicy:
        policy = self.policies.get((axis, stratum)) or self.policies.get((axis, "*"))
        if policy is None:
            raise KeyError(f"no debt policy for axis={axis}, stratum={stratum}")
        return policy

    def add_omission(
        self,
        *,
        axis: str,
        stratum: str,
        item_key: str,
        created_case: int,
        estimated_units: float = 1.0,
        debt_kind: str = "quota",
        exact_replayable: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> CoverageDebtEntry:
        self._policy(axis, stratum)
        serial = self._next_serial
        self._next_serial += 1
        identity = {
            "schema_version": "rlcmf-coverage-debt-id-v1",
            "manifest_sha256": self.manifest_sha256,
            "serial": serial,
            "axis": axis,
            "stratum": stratum,
            "item_key": item_key,
            "created_case": int(created_case),
        }
        debt_id = hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()
        entry = CoverageDebtEntry(
            debt_id=debt_id,
            axis=str(axis),
            stratum=str(stratum),
            item_key=str(item_key),
            created_case=int(created_case),
            estimated_units=float(estimated_units),
            debt_kind=str(debt_kind),
            exact_replayable=bool(exact_replayable),
            metadata=dict(metadata or {}),
        )
        self.entries[debt_id] = entry
        return entry

    def add_structured_omissions(
        self,
        *,
        axis: str,
        base_stratum: CoverageStratum | Mapping[str, Any],
        omitted_items: Mapping[str, Iterable[str]],
        opportunity_id: str,
        created_case: int,
        estimated_units_per_item: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[CoverageDebtEntry]:
        """Create one independently aged quota debt record per omitted capability."""

        if axis not in AUDIT_AXES:
            raise ValueError(f"unknown debt axis: {axis}")
        normalized_opportunity = str(opportunity_id or "").strip()
        if not normalized_opportunity:
            raise ValueError("structured omission opportunity_id cannot be empty")
        base = (
            base_stratum
            if isinstance(base_stratum, CoverageStratum)
            else CoverageStratum.from_payload(base_stratum)
        )
        normalized_items: list[tuple[str, str]] = []
        for dimension, values in omitted_items.items():
            normalized_dimension = str(dimension or "").strip()
            if normalized_dimension not in COVERAGE_STRATUM_DIMENSIONS:
                raise ValueError(
                    f"unknown coverage stratum dimension: {normalized_dimension}"
                )
            for value in values:
                normalized_value = str(value or "").strip()
                if not normalized_value:
                    raise ValueError("omitted coverage item cannot be empty")
                normalized_items.append((normalized_dimension, normalized_value))
        normalized_items = sorted(set(normalized_items))
        if not normalized_items:
            raise ValueError("structured omission requires at least one item")
        entries = []
        for dimension, value in normalized_items:
            item_stratum = base.with_item(dimension, value)
            item_key = f"{normalized_opportunity}:{dimension}={value}"
            item_metadata = {
                **dict(metadata or {}),
                "coverage_stratum": item_stratum.to_dict(),
                "coverage_item_dimension": dimension,
                "coverage_item_value": value,
                "opportunity_id": normalized_opportunity,
            }
            entries.append(
                self.add_omission(
                    axis=axis,
                    stratum=item_stratum.key,
                    item_key=item_key,
                    created_case=created_case,
                    estimated_units=estimated_units_per_item,
                    debt_kind="quota",
                    exact_replayable=False,
                    metadata=item_metadata,
                )
            )
        return entries

    def outstanding(
        self,
        *,
        axis: str | None = None,
        stratum: str | None = None,
        debt_kind: str | None = None,
    ) -> list[CoverageDebtEntry]:
        entries = [entry for entry in self.entries.values() if entry.status == "outstanding"]
        if axis is not None:
            entries = [entry for entry in entries if entry.axis == axis]
        if stratum is not None:
            entries = [entry for entry in entries if entry.stratum == stratum]
        if debt_kind is not None:
            if debt_kind not in {"quota", "exact"}:
                raise ValueError("debt_kind must be quota or exact")
            entries = [entry for entry in entries if entry.debt_kind == debt_kind]
        return sorted(entries, key=lambda entry: (entry.created_case, entry.debt_id))

    def forced_settlements(self, *, current_case: int) -> list[ForcedSettlement]:
        if int(current_case) < 0:
            raise ValueError("current_case must be non-negative")
        requirements: dict[str, set[str]] = {}
        grouped: dict[tuple[str, str], list[CoverageDebtEntry]] = {}
        for entry in self.outstanding():
            grouped.setdefault((entry.axis, entry.stratum), []).append(entry)
        for key, entries in grouped.items():
            policy = self._policy(*key)
            for entry in entries:
                if entry.age(current_case) >= policy.max_age_cases:
                    requirements.setdefault(entry.debt_id, set()).add("age_cap")
            overflow = max(0, len(entries) - policy.max_outstanding)
            for entry in entries[:overflow]:
                requirements.setdefault(entry.debt_id, set()).add("outstanding_cap")
        settlements = [
            ForcedSettlement(
                debt_id=entry.debt_id,
                axis=entry.axis,
                stratum=entry.stratum,
                item_key=entry.item_key,
                reason="+".join(sorted(reasons)),
                age_cases=entry.age(current_case),
                exact_replayable=entry.exact_replayable,
            )
            for debt_id, reasons in sorted(requirements.items())
            if (entry := self.entries.get(debt_id)) is not None
        ]
        return sorted(
            settlements,
            key=lambda item: (
                self.entries[item.debt_id].created_case,
                item.debt_id,
            ),
        )

    def settle(
        self,
        debt_id: str,
        *,
        settled_case: int,
        reason: str,
        audit_decision_id: str = "",
    ) -> CoverageDebtEntry:
        entry = self.entries.get(str(debt_id))
        if entry is None:
            raise KeyError(f"unknown coverage debt: {debt_id}")
        if entry.status != "outstanding":
            raise ValueError(f"coverage debt already settled: {debt_id}")
        if int(settled_case) < entry.created_case:
            raise ValueError("settled_case cannot precede created_case")
        if not str(reason or "").strip():
            raise ValueError("settlement reason cannot be empty")
        if audit_decision_id:
            _validated_sha256(
                str(audit_decision_id),
                field="audit_decision_id",
            )
        entry.status = "settled"
        entry.settled_case = int(settled_case)
        entry.settlement_reason = str(reason)
        entry.audit_decision_id = str(audit_decision_id or "")
        return entry

    def settle_oldest(
        self,
        *,
        axis: str,
        stratum: str,
        settled_case: int,
        reason: str,
        audit_decision_id: str = "",
        debt_kind: str | None = None,
    ) -> CoverageDebtEntry | None:
        entries = self.outstanding(
            axis=axis,
            stratum=stratum,
            debt_kind=debt_kind,
        )
        if not entries:
            return None
        return self.settle(
            entries[0].debt_id,
            settled_case=settled_case,
            reason=reason,
            audit_decision_id=audit_decision_id,
        )

    def summary(self, *, current_case: int) -> dict[str, Any]:
        outstanding = self.outstanding()
        by_stratum: dict[str, dict[str, Any]] = {}
        by_item: dict[str, dict[str, Any]] = {}
        for entry in outstanding:
            key = f"{entry.axis}:{entry.stratum}"
            row = by_stratum.setdefault(
                key,
                {
                    "axis": entry.axis,
                    "stratum": entry.stratum,
                    "outstanding_count": 0,
                    "outstanding_units": 0.0,
                    "oldest_age_cases": 0,
                    "exact_replayable_count": 0,
                },
            )
            row["outstanding_count"] += 1
            row["outstanding_units"] += entry.estimated_units
            row["oldest_age_cases"] = max(
                row["oldest_age_cases"],
                entry.age(current_case),
            )
            row["exact_replayable_count"] += int(entry.exact_replayable)
            item_dimension = str(
                entry.metadata.get("coverage_item_dimension", "") or ""
            )
            item_value = str(entry.metadata.get("coverage_item_value", "") or "")
            if item_dimension and item_value:
                item_summary_key = f"{entry.axis}:{item_dimension}={item_value}"
                item_row = by_item.setdefault(
                    item_summary_key,
                    {
                        "axis": entry.axis,
                        "dimension": item_dimension,
                        "value": item_value,
                        "outstanding_count": 0,
                        "outstanding_units": 0.0,
                        "oldest_age_cases": 0,
                    },
                )
                item_row["outstanding_count"] += 1
                item_row["outstanding_units"] += entry.estimated_units
                item_row["oldest_age_cases"] = max(
                    item_row["oldest_age_cases"],
                    entry.age(current_case),
                )
        forced = self.forced_settlements(current_case=current_case)
        return {
            "schema_version": "rlcmf-coverage-debt-summary-v1",
            "manifest_sha256": self.manifest_sha256,
            "current_case": int(current_case),
            "entry_count": len(self.entries),
            "outstanding_count": len(outstanding),
            "settled_count": len(self.entries) - len(outstanding),
            "outstanding_units": sum(entry.estimated_units for entry in outstanding),
            "forced_settlement_count": len(forced),
            "forced_settlements": [item.to_dict() for item in forced],
            "by_stratum": dict(sorted(by_stratum.items())),
            "by_item": dict(sorted(by_item.items())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rlcmf-coverage-debt-ledger-v1",
            "manifest_sha256": self.manifest_sha256,
            "next_serial": self._next_serial,
            "policies": [
                policy.to_dict()
                for policy in sorted(
                    self.policies.values(),
                    key=lambda item: item.key,
                )
            ],
            "entries": [
                entry.to_dict()
                for entry in sorted(
                    self.entries.values(),
                    key=lambda item: (item.created_case, item.debt_id),
                )
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CoverageDebtLedger":
        if payload.get("schema_version") != "rlcmf-coverage-debt-ledger-v1":
            raise ValueError("unsupported coverage-debt ledger schema")
        raw_policies = payload.get("policies", [])
        if not isinstance(raw_policies, list):
            raise ValueError("coverage-debt policies must be a list")
        policies = [
            CoverageDebtPolicy(
                axis=str(item["axis"]),
                stratum=str(item["stratum"]),
                max_age_cases=int(item["max_age_cases"]),
                max_outstanding=int(item["max_outstanding"]),
            )
            for item in raw_policies
            if isinstance(item, Mapping)
        ]
        if len(policies) != len(raw_policies):
            raise ValueError("invalid coverage-debt policy payload")
        ledger = cls(
            manifest_sha256=str(payload.get("manifest_sha256", "")),
            policies=policies,
        )
        raw_entries = payload.get("entries", [])
        if not isinstance(raw_entries, list):
            raise ValueError("coverage-debt entries must be a list")
        for item in raw_entries:
            if not isinstance(item, Mapping):
                raise ValueError("invalid coverage-debt entry payload")
            entry = CoverageDebtEntry(
                debt_id=_validated_sha256(str(item["debt_id"]), field="debt_id"),
                axis=str(item["axis"]),
                stratum=str(item["stratum"]),
                item_key=str(item["item_key"]),
                created_case=int(item["created_case"]),
                estimated_units=float(item["estimated_units"]),
                debt_kind=str(item["debt_kind"]),
                exact_replayable=bool(item["exact_replayable"]),
                metadata=dict(item.get("metadata", {}) or {}),
                status=str(item.get("status", "outstanding")),
                settled_case=(
                    None
                    if item.get("settled_case") is None
                    else int(item["settled_case"])
                ),
                settlement_reason=str(item.get("settlement_reason", "")),
                audit_decision_id=str(item.get("audit_decision_id", "")),
            )
            if entry.debt_id in ledger.entries:
                raise ValueError(f"duplicate coverage debt id: {entry.debt_id}")
            ledger._policy(entry.axis, entry.stratum)
            ledger.entries[entry.debt_id] = entry
        ledger._next_serial = int(payload.get("next_serial", len(ledger.entries)))
        if ledger._next_serial < len(ledger.entries):
            raise ValueError("next_serial cannot be smaller than entry count")
        return ledger
