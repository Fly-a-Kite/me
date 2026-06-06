from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from datadiff.datagen import repair_operations
from datadiff.dsl import Case, Program
from datadiff.util import JsonlWriter, RUNS_DIR, append_jsonl, iter_jsonl, utc_now

CHAMPION_CORPUS_SCHEMA_VERSION = "champion-corpus-v1"
DEFAULT_CHAMPION_CORPUS_PATH = RUNS_DIR / "champion_corpus.jsonl"


@dataclass(frozen=True, slots=True)
class ChampionSeed:
    case_id: str
    version_id: str
    bug_family_keys: tuple[str, ...]
    case_payload: dict[str, Any]
    minhash_signature: tuple[int, ...] = ()
    promoted_at: str = ""
    stability: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CHAMPION_CORPUS_SCHEMA_VERSION,
            "case_id": self.case_id,
            "version_id": self.version_id,
            "bug_family_keys": list(self.bug_family_keys),
            "case_payload": copy.deepcopy(self.case_payload),
            "minhash_signature": list(self.minhash_signature),
            "promoted_at": self.promoted_at,
            "stability": self.stability,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChampionSeed":
        return cls(
            case_id=str(data.get("case_id", "") or ""),
            version_id=str(data.get("version_id", "") or ""),
            bug_family_keys=tuple(_unique_strings(data.get("bug_family_keys", []) or [])),
            case_payload=copy.deepcopy(dict(data.get("case_payload", {}) or {})),
            minhash_signature=tuple(
                int(value)
                for value in data.get("minhash_signature", []) or []
                if _looks_int(value)
            ),
            promoted_at=str(data.get("promoted_at", "") or ""),
            stability=max(0, int(data.get("stability", 0) or 0)),
        )

    def case(self) -> Case:
        case = Case.from_dict(copy.deepcopy(self.case_payload))
        metadata = dict(case.metadata or {})
        metadata["champion_seed"] = {
            "source_version_id": self.version_id,
            "bug_family_keys": list(self.bug_family_keys),
            "stability": self.stability,
            "promoted_at": self.promoted_at,
        }
        case.metadata = metadata
        return case

    def identity_key(self) -> tuple[str, str, tuple[str, ...]]:
        return (self.version_id, self.case_id, self.bug_family_keys)


@dataclass(slots=True)
class ChampionRegistry:
    path: Path = DEFAULT_CHAMPION_CORPUS_PATH

    def promote_if_stable(
        self,
        case: Case,
        family_keys: list[str],
        threshold: int = 3,
        *,
        version_id: str = "",
        stability: int | None = None,
    ) -> bool:
        families = tuple(_unique_strings(family_keys))
        if not families:
            return False
        observed_stability = max(0, int(stability if stability is not None else threshold))
        if observed_stability < max(1, int(threshold)):
            return False
        champion = ChampionSeed(
            case_id=str(case.case_id),
            version_id=str(version_id or ""),
            bug_family_keys=families,
            case_payload=copy.deepcopy(case.to_dict()),
            minhash_signature=_case_minhash_signature(case),
            promoted_at=utc_now(),
            stability=observed_stability,
        )
        existing = self._read_all()
        for index, stored in enumerate(existing):
            if stored.identity_key() != champion.identity_key():
                continue
            if stored.stability >= champion.stability:
                return False
            existing[index] = champion
            self._write_all(existing)
            return True
        append_jsonl(champion.to_dict(), self.path)
        return True

    def champions_for_version(
        self,
        version_id: str,
        *,
        include_same_version: bool = False,
        limit: int | None = None,
    ) -> list[ChampionSeed]:
        resolved_version = str(version_id or "")
        champions = []
        for champion in self._read_all():
            if not champion.case_id or not champion.case_payload:
                continue
            if resolved_version and not include_same_version and champion.version_id == resolved_version:
                continue
            champions.append(champion)
        champions.sort(
            key=lambda item: (
                item.stability,
                len(item.bug_family_keys),
                item.promoted_at,
                item.case_id,
            ),
            reverse=True,
        )
        if limit is None:
            return champions
        return champions[: max(0, int(limit))]

    def graft_subtree(self, host: Case, donor: ChampionSeed, rnd: random.Random) -> Case:
        donor_case = donor.case()
        host_payload = host.to_dict()
        host_tables = Case.from_dict(host_payload).tables
        host_operations = [copy.deepcopy(operation) for operation in host.program.operations]
        donor_operations = [copy.deepcopy(operation) for operation in donor_case.program.operations]
        if donor_operations:
            width = rnd.randint(1, min(3, len(donor_operations)))
            start = rnd.randint(0, max(0, len(donor_operations) - width))
            grafted = donor_operations[start : start + width]
            operations = repair_operations(
                host_tables[0],
                [*host_operations, *grafted],
                extra_tables=host_tables[1:],
            )
        else:
            grafted = []
            operations = list(host_operations)
        if not operations:
            operations = list(host_operations) or [{"op": "limit", "n": len(host_tables[0].rows)}]
        metadata = dict(host.metadata or {})
        metadata["champion_graft"] = {
            "donor_case_id": donor.case_id,
            "donor_version_id": donor.version_id,
            "bug_family_keys": list(donor.bug_family_keys),
            "grafted_operation_count": len(grafted),
        }
        return Case(
            case_id=f"{host.case_id}-champion-graft-{rnd.randrange(1_000_000)}",
            seed=host.seed,
            tables=host_tables,
            program=Program(
                program_id=f"{host.program.program_id}-champion-graft",
                seed=host.program.seed,
                operations=operations,
            ),
            metadata=metadata,
        )

    def _read_all(self) -> list[ChampionSeed]:
        if not self.path.exists():
            return []
        champions: dict[tuple[str, str, tuple[str, ...]], ChampionSeed] = {}
        for row in iter_jsonl(self.path):
            if not isinstance(row, dict):
                continue
            try:
                champion = ChampionSeed.from_dict(row)
            except Exception:
                continue
            key = champion.identity_key()
            stored = champions.get(key)
            if stored is None or champion.stability > stored.stability:
                champions[key] = champion
        return list(champions.values())

    def _write_all(self, champions: Iterable[ChampionSeed]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with JsonlWriter(self.path, mode="wt", sort_keys=True) as writer:
            for champion in champions:
                writer.write(champion.to_dict())


def champion_signature(champion: ChampionSeed) -> str:
    digest = hashlib.sha1(
        "|".join(
            [
                champion.version_id,
                champion.case_id,
                ",".join(champion.bug_family_keys),
                ",".join(str(value) for value in champion.minhash_signature[:16]),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return f"champion:{digest}"


def _case_minhash_signature(case: Case) -> tuple[int, ...]:
    metadata = case.metadata if isinstance(case.metadata, dict) else {}
    fingerprint = metadata.get("case_fingerprint", {})
    if not isinstance(fingerprint, dict):
        return ()
    signature = fingerprint.get("minhash_signature", [])
    if not isinstance(signature, list):
        return ()
    return tuple(int(value) for value in signature if _looks_int(value))


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _looks_int(value: Any) -> bool:
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True
