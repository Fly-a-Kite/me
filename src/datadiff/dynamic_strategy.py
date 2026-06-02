from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

from datadiff.util import PROJECT_ROOT
from datadiff.util import REPORTS_DIR
from datadiff.util import dump_json
from datadiff.util import load_json
from datadiff.util import slugify
from datadiff.util import utc_now


STRATEGY_SNAPSHOT_SCHEMA_VERSION = "dynamic-strategy-snapshot-v1"
DEFAULT_STRATEGY_SNAPSHOT_DIR = REPORTS_DIR / "strategy-snapshots"
DEFAULT_STRATEGY_LEARNING_DIR = REPORTS_DIR / "strategy-learning"


@dataclass(frozen=True, slots=True)
class StrategyRuleRecord:
    rule_id: str
    kind: str
    reason: str
    priority: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StrategySnapshot:
    snapshot_id: str
    generated_at: str
    classification_documented_rules: tuple[StrategyRuleRecord, ...] = ()
    classification_boundary_rules: tuple[StrategyRuleRecord, ...] = ()
    reproducer_rules: tuple[StrategyRuleRecord, ...] = ()
    learning_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STRATEGY_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "generated_at": self.generated_at,
            "classification_documented_rules": [rule.to_dict() for rule in self.classification_documented_rules],
            "classification_boundary_rules": [rule.to_dict() for rule in self.classification_boundary_rules],
            "reproducer_rules": [rule.to_dict() for rule in self.reproducer_rules],
            "learning_summary": dict(self.learning_summary),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategySnapshot":
        def _records(key: str) -> tuple[StrategyRuleRecord, ...]:
            values = payload.get(key, [])
            if not isinstance(values, list):
                return ()
            rows: list[StrategyRuleRecord] = []
            for value in values:
                if not isinstance(value, dict):
                    continue
                rows.append(
                    StrategyRuleRecord(
                        rule_id=str(value.get("rule_id", "")),
                        kind=str(value.get("kind", "")),
                        reason=str(value.get("reason", "")),
                        priority=int(value.get("priority", 100) or 100),
                        metadata=dict(value.get("metadata", {}) or {}),
                    )
                )
            return tuple(rows)

        return cls(
            snapshot_id=str(payload.get("snapshot_id", "")),
            generated_at=str(payload.get("generated_at", "")),
            classification_documented_rules=_records("classification_documented_rules"),
            classification_boundary_rules=_records("classification_boundary_rules"),
            reproducer_rules=_records("reproducer_rules"),
            learning_summary=dict(payload.get("learning_summary", {}) or {}),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def write_strategy_snapshot(
    *,
    classification_documented_rules: list[StrategyRuleRecord] | tuple[StrategyRuleRecord, ...],
    classification_boundary_rules: list[StrategyRuleRecord] | tuple[StrategyRuleRecord, ...],
    reproducer_rules: list[StrategyRuleRecord] | tuple[StrategyRuleRecord, ...],
    learning_summary: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    snapshot_id: str | None = None,
) -> Path:
    resolved_dir = Path(output_dir or DEFAULT_STRATEGY_SNAPSHOT_DIR)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    generated_at = utc_now()
    resolved_snapshot_id = snapshot_id or slugify(f"strategy-snapshot-{generated_at}", max_len=120)
    path = resolved_dir / f"{resolved_snapshot_id}.json"
    snapshot = StrategySnapshot(
        snapshot_id=resolved_snapshot_id,
        generated_at=generated_at,
        classification_documented_rules=tuple(classification_documented_rules),
        classification_boundary_rules=tuple(classification_boundary_rules),
        reproducer_rules=tuple(reproducer_rules),
        learning_summary=dict(learning_summary or {}),
        metadata=dict(metadata or {}),
    )
    dump_json(snapshot.to_dict(), path)
    return path


def load_strategy_snapshot(path: str | Path | None) -> StrategySnapshot | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        candidates = [
            resolved,
            PROJECT_ROOT / resolved,
            DEFAULT_STRATEGY_SNAPSHOT_DIR / resolved,
        ]
        resolved = next((candidate for candidate in candidates if candidate.is_file()), candidates[-1])
    if not resolved.is_file():
        return None
    payload = load_json(resolved)
    if not isinstance(payload, dict):
        return None
    if str(payload.get("schema_version", "")) != STRATEGY_SNAPSHOT_SCHEMA_VERSION:
        return None
    return StrategySnapshot.from_dict(payload)


def append_learning_event(
    event: dict[str, Any],
    *,
    output_dir: Path | None = None,
    learning_id: str = "strategy-learning",
) -> Path:
    resolved_dir = Path(output_dir or DEFAULT_STRATEGY_LEARNING_DIR)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    path = resolved_dir / f"{slugify(learning_id, max_len=100)}.json"
    existing = load_json(path) if path.is_file() else {"schema_version": "dynamic-strategy-learning-v1", "events": []}
    if not isinstance(existing, dict):
        existing = {"schema_version": "dynamic-strategy-learning-v1", "events": []}
    events = existing.get("events", [])
    if not isinstance(events, list):
        events = []
    payload = dict(event)
    payload.setdefault("recorded_at", utc_now())
    events.append(payload)
    existing["events"] = events
    dump_json(existing, path)
    return path
